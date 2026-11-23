"""Contextual detection of requests for a specific person's private or secret data.

A keyword alone is not evidence. "What is an API key?" and "What is the API key for my
account?" share the keyword but differ in what is being asked. A question is treated as
a private-data request only when it combines:

1. a SENSITIVE ITEM (credential, financial / government identifier, contact details,
   medical record, private communication, personal record),
2. a SPECIFIC OWNER: the asker ("my", "mine", "our", "for my account") or a specific
   private third party reached through the asker ("my mother's", "my neighbour"), and
3. RETRIEVAL INTENT: the asker wants the value itself ("what is", "tell me", "give me",
   "how much was"), not an explanation or a procedure.

Conceptual questions ("what is a/an …", "how do … work", "explain", "why should …") and
procedures ("how do I reset/change/protect my password") are never flagged. The model has
no access to such data, so a flagged request is epistemically UNKNOWN regardless of what
the model generates.

Input is normalised first so that light obfuscation ("p-a-s-s-w-o-r-d", "P I N"), line
breaks, markdown and JSON wrapping do not hide the request.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional

# --------------------------------------------------------------------------------------
# Sensitive item lexicon, grouped by category. Each entry is a regex fragment matched on
# normalised, lower-cased text.
# --------------------------------------------------------------------------------------
SENSITIVE_ITEMS = {
    "credential": [
        r"pass(word|phrase|code)s?", r"pins?(\s+(code|number))?", r"(lock\s+|safe\s+)?combinations?",
        r"api\s+keys?", r"(wi-?fi|network|wpa2?|router|license|licence|product|encryption)\s+keys?", r"access\s+tokens?", r"auth(entication)?\s+tokens?", r"secret\s+keys?",
        r"private\s+keys?", r"seed\s+phrases?", r"recovery\s+(codes?|phrases?|keys?)",
        r"(otp|2fa|one[\s-]time|verification|security)\s+codes?", r"security\s+answers?",
        r"(login|sign[\s-]in)\s+(details|credentials)", r"credentials?",
    ],
    "financial": [
        r"(bank\s+)?account\s+numbers?", r"routing\s+numbers?", r"sort\s+codes?", r"iban",
        r"(credit|debit)\s+card(\s+(numbers?|details))?", r"card\s+numbers?", r"cvv2?", r"cvc", r"security\s+codes?",
        r"salary", r"income", r"(bank|account|statement|card|savings|checking)\s+balance", r"tax\s+returns?", r"net\s+worth",
    ],
    "government_id": [
        r"social\s+security\s+numbers?", r"ssn", r"passport\s+numbers?", r"driv(er'?s|ing)\s+licen[cs]e(\s+numbers?)?",
        r"national\s+(id|insurance)(\s+numbers?)?", r"aadhaar(\s+numbers?)?", r"tax\s+id(\s+numbers?)?",
    ],
    "contact_location": [
        r"e-?mail\s+address(es)?", r"(mobile|cell|phone|telephone)\s+numbers?",
        r"(?<!\bip\s)(?<!\bmac\s)(?<!\bweb\s)(?<!\burl\s)(home|street|postal|mailing)?\s*address(es)?",
        r"location", r"whereabouts",
    ],
    "medical": [
        r"medications?", r"prescriptions?", r"diagnos(is|es)", r"medical\s+(records?|history|conditions?)",
        r"test\s+results?", r"blood\s+type",
    ],
    "private_record": [
        r"diary", r"journal", r"notebook", r"private\s+(messages?|chats?|conversations?|meetings?|calls?|notes?)",
        r"(text|whatsapp|direct)\s+messages?", r"messages?", r"texts", r"voicemails?", r"dms?", r"emails?",
        r"browsing\s+history", r"search\s+history",
    ],
}

_ITEM_RES = {cat: re.compile(r"\b(" + "|".join(items) + r")\b") for cat, items in SENSITIVE_ITEMS.items()}

# Owner: the asker, or a specific person reached through the asker.
_RELATION = (
    r"((ex-?)?(mother|mom|mum|father|dad|parents?|wife|husband|partner|spouse|girlfriend|boyfriend)|son|daughter|"
    r"brother|sister|sibling|friend|neighbou?r|boss|manager|colleague|co-?worker|roommate|landlord|"
    r"teacher|doctor|child|children|kids?|family|grandmother|grandfather|aunt|uncle|cousin)"
)
_OWNER_RE = re.compile(
    r"\b(my|mine|our|ours)\b"                     # first person possession
    r"|\bmy\s+" + _RELATION + r"('s)?\b"         # my mother's …
    r"|\b(i|we)\s+(have|had|use|used|set|wrote|stored|saved|keep|kept|signed\s+up|registered|entered|created|chose|picked|gave)\b"  # the password I set
    r"|\b(did|do|have|had)\s+(i|we)\s+(use|set|sign\s+up|register|enter|create|choose|pick|give|save|store)\b"  # which email did I sign up with
    r"|\b(am|was|are|were)\s+(i|we)\s+(currently\s+|still\s+)?(prescribed|taking|on|using|enrolled|registered|paid|earning)\b"  # medications am I prescribed
)

# Retrieval intent: asking for the value itself.
_RETRIEVAL_RE = re.compile(
    r"\b(what|what's|whats|which)\b|\btell\s+me\b|\bgive\s+me\b|\bshow\s+me\b|\bsend\s+me\b|\blist\b|"
    r"\breveal\b|\bfind\b|\blook\s+up\b|\bretrieve\b|\bfetch\b|\bpull\s+up\b|\bget\s+me\b|\bprovide\s+me\b|\bread\s+out\b|\bremind\s+me\b|\bdo\s+you\s+know\b|\bhow\s+much\b|\bhow\s+many\b|"
    r"\bwhere\s+(is|are|do|does|did)\b|\bwho\s+(is|are)\b|\brecall\b|\bguess\b|\bread\s+(me|out)\b"
)

# "What did my wife text me?" — content of a specific private communication.
_PRIVATE_COMMUNICATION_RE = re.compile(
    r"\bwhat\s+did\s+my\s+[a-z-]+\s+(text|message|email|e-mail|tell|say|write|whisper|send)\b"
)

# "Where does my sister live?" — a specific private person's location.
_PERSON_LOCATION_RE = re.compile(
    r"\bwhere\s+(does|do|did|is)\s+my\s+" + r"((ex-?)?[a-z-]+)" + r"\s+(live|lives|stay|work)\b"
)

# Conceptual or procedural framing: the asker wants an explanation or a method, not the value.
_CONCEPTUAL_RE = re.compile(
    r"^\s*(what|which)\s+(is|are)\s+(a|an)\b"                                   # what is a PIN
    r"|^\s*(what|which)\s+(is|are)\s+the\s+(purpose|point|format|meaning|difference|definition|role|use|length|structure)\b"
    r"|\bhow\s+(do|does|can|should|would|could|to)\s+(i|you|we|one|people|it|they|[a-z]+s)?\s*"
    r"(reset|change|recover|retrieve|fetch|obtain|request|create|choose|pick|protect|secure|store|manage|update|generate|rotate|revoke|"
    r"replace|block|cancel|unlock|apply\s+for|find|see|check|view|look\s+up|get|access|work)\b"
    r"|\bhow\s+(do|does)\b.*\bwork\b"
    r"|\bwhat\s+(happens|should\s+i\s+do)\s+(if|when)\b"
    r"|\bwhere\s+(can|do|should|would)\s+(i|you|we|one)\s+(find|see|check|get|look|view|locate|change|reset)\b"
    r"|\b(what|which)('s|\s+is|\s+are)\s+(the\s+)?(best|safest|easiest|right|correct|recommended|proper)\s+(way|ways|method|practice)s?\b"
    r"|^\s*(can|could|may|will|would|does|do|is|are)\s+(my\s+)?[a-z]+\s+(read|see|access|view|use|track|share)\b"
    r"|\b(explain|describe|define)\b"
    r"|^\s*why\s+(should|do|does|is|are|would)\b"
    r"|\bwhat\s+makes\b|\bbest\s+practices?\b|\bis\s+it\s+safe\s+to\b|\bshould\s+i\b"
)


def normalize_text(text: str) -> str:
    """Lower-case and undo light obfuscation so patterns see the intended words.

    * Unicode NFKC; markdown emphasis/code markers and JSON punctuation become spaces.
    * Line breaks and repeated whitespace collapse to single spaces.
    * Runs of single characters separated by spaces, dots, hyphens or underscores are
      re-joined ("p-a-s-s-w-o-r-d" -> "password", "P I N" -> "pin").
    """
    t = unicodedata.normalize("NFKC", text or "").lower()
    t = t.replace("’", "'")
    t = re.sub(r"[*_`#>{}\[\]\"|]", " ", t)
    t = re.sub(r"\s+", " ", t)

    def _join(m: re.Match) -> str:
        return re.sub(r"[\s.\-_]", "", m.group(0))

    # two or more single letters separated by one separator char, e.g. "p a s s" / "p-i-n"
    t = re.sub(r"\b(?:[a-z][\s.\-_]){2,}[a-z]\b", _join, t)
    return t.strip()


@dataclass
class SensitiveRequestAssessment:
    is_private_request: bool
    categories: List[str] = field(default_factory=list)
    items: List[str] = field(default_factory=list)
    owner: Optional[str] = None
    reason: str = ""
    evidence: List[str] = field(default_factory=list)


def is_conceptual_question(text: str) -> bool:
    """True for explanations / procedures ("what is a PIN", "how do I reset my password")."""
    return any(_CONCEPTUAL_RE.search(c) for c in (_clauses(normalize_text(text)) or [""]))


def _clauses(text: str) -> List[str]:
    """Split into sentence-like clauses so cues from different sentences don't combine."""
    parts = re.split(r"(?<=[?.!;:])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


class SensitiveRequestDetector:
    """Detects requests for a specific person's private or secret data (see module doc)."""

    @classmethod
    def assess(cls, question: str) -> SensitiveRequestAssessment:
        norm = normalize_text(question)
        for clause in _clauses(norm) or [norm]:
            res = cls._assess_clause(clause)
            if res.is_private_request:
                return res
        return SensitiveRequestAssessment(False, reason="No request for a specific person's private data.")

    @classmethod
    def _assess_clause(cls, clause: str) -> SensitiveRequestAssessment:
        cats, items = [], []
        for cat, rx in _ITEM_RES.items():
            for m in rx.finditer(clause):
                if m.group(0).strip():
                    cats.append(cat)
                    items.append(m.group(0).strip())
        comm = _PRIVATE_COMMUNICATION_RE.search(clause)
        if comm:
            cats.append("private_record")
            items.append(comm.group(0))
        loc = _PERSON_LOCATION_RE.search(clause)
        if loc:
            cats.append("contact_location")
            items.append(loc.group(0))
        if not items:
            return SensitiveRequestAssessment(False)
        owner = _OWNER_RE.search(clause)
        if not owner:
            return SensitiveRequestAssessment(False, sorted(set(cats)), items, reason="Sensitive term without a specific owner.")
        if _CONCEPTUAL_RE.search(clause):
            return SensitiveRequestAssessment(False, sorted(set(cats)), items, owner.group(0),
                                              reason="Conceptual or procedural question about a sensitive topic.")
        if not _RETRIEVAL_RE.search(clause):
            return SensitiveRequestAssessment(False, sorted(set(cats)), items, owner.group(0),
                                              reason="No request for the value itself.")
        cats_u = sorted(set(cats))
        return SensitiveRequestAssessment(
            True, cats_u, items, owner.group(0),
            reason=f"Requests the value of {', '.join(dict.fromkeys(items))} belonging to a specific person ('{owner.group(0)}').",
            evidence=[f"private_{c}" for c in cats_u] + [f"owner:{owner.group(0)}"],
        )
