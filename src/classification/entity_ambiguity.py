"""Context-aware entity (polysemy) ambiguity assessment.

Replaces bare keyword matching ("the question contains 'jordan', so it is ambiguous") with
a check of whether the *question* already selects one sense. A polysemous term is
ambiguous only when none of the following resolve it:

1. **Sense cue**: the question names a sense or a fact unique to it
   ("the planet Mercury", "the US state of Georgia", "Michael Jordan the basketball player").
2. **Adjacent proper name**: the term is part of a longer proper name
   ("Michael Jordan", "Jordan Peterson", "Freddie Mercury", "George Washington").
3. **Sense-restricting attribute**: the question asks for an attribute that only one
   sense has ("capital of Jordan" — only the country has a capital; "atomic number of
   mercury" — only the element). An attribute shared by several senses ("capital of
   Georgia": the country and the US state both have one) leaves the term ambiguous.

The inventory covers the polysemous entities the pipeline already tracked
(``KNOWN_POLYSEMIC_TERMS`` / ``POLYSEMIC_TERMS``); it encodes *senses*, not answers.
Unknown polysemous words (e.g. "Java", "bat") are not covered — see
docs/KNOWN_LIMITATIONS.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# term -> {sense: {"cues": [...], "attributes": [...]}}
SENSE_INVENTORY: Dict[str, Dict[str, Dict[str, List[str]]]] = {
    "georgia": {
        "country": {"cues": [r"country", r"nation", r"republic", r"caucasus", r"tbilisi", r"georgian\s+(language|lari)", r"black\s+sea", r"former\s+soviet"],
                    "attributes": [r"capital", r"population", r"flag", r"currency", r"president", r"prime\s+minister", r"language", r"area", r"history"]},
        "us_state": {"cues": [r"(us|u\.s\.|american|united\s+states)\s+state", r"usa\b", r"u\.s\.a?\.?(?=\W|$)", r"georgia,?\s+us\b", r"state\s+of", r"\bstate\b", r"atlanta", r"savannah", r"peach\s+state", r"governor"],
                     "attributes": [r"capital", r"population", r"flag", r"governor", r"area", r"history", r"county", r"counties"]},
    },
    "jordan": {
        "country": {"cues": [r"country", r"kingdom", r"hashemite", r"amman", r"middle\s+east", r"petra", r"dead\s+sea", r"king\s+abdullah"],
                    "attributes": [r"capital", r"population", r"currency", r"flag", r"king", r"border", r"official\s+language", r"gdp"]},
        "person": {"cues": [r"basketball", r"nba", r"bulls", r"actor", r"psycholog", r"author", r"player", r"singer"],
                   "attributes": [r"born", r"age", r"married", r"wife", r"career", r"height", r"net\s+worth", r"movies?", r"books?"]},
        "river": {"cues": [r"river", r"jordan\s+valley"], "attributes": [r"flows?", r"length", r"source"]},
    },
    "mercury": {
        "planet": {"cues": [r"planet", r"orbit", r"solar\s+system", r"from\s+the\s+sun"],
                   "attributes": [r"orbit", r"distance", r"diameter", r"moons?", r"day\s+length", r"surface\s+temperature", r"gravity"]},
        "element": {"cues": [r"element", r"chemical", r"\bhg\b", r"metal", r"thermometer", r"toxic", r"poison"],
                    "attributes": [r"atomic\s+(number|mass|weight)", r"boiling\s+point", r"melting\s+point", r"density", r"toxicity", r"symbol"]},
        "god": {"cues": [r"\bgod\b", r"roman", r"myth"], "attributes": [r"greek\s+equivalent", r"worship"]},
        "person": {"cues": [r"freddie", r"queen\s+(band|singer)", r"singer"], "attributes": [r"songs?", r"born", r"died"]},
    },
    "washington": {
        "us_state": {"cues": [r"state", r"seattle", r"olympia", r"pacific\s+northwest"], "attributes": [r"governor", r"county", r"counties"]},
        "dc": {"cues": [r"d\.?\s?c\.?", r"district\s+of\s+columbia", r"us\s+capital", r"white\s+house"], "attributes": [r"mayor", r"monuments?"]},
        "person": {"cues": [r"george", r"president", r"general", r"denzel", r"actor"], "attributes": [r"born", r"died", r"presiden(cy|t)", r"teeth"]},
    },
    "jaguar": {
        "animal": {"cues": [r"animal", r"cat", r"feline", r"wild", r"rainforest", r"predator", r"species"], "attributes": [r"habitat", r"diet", r"eat", r"lifespan", r"weigh"]},
        "car_brand": {"cues": [r"\bcars?\b", r"automobile", r"vehicle", r"brand", r"land\s+rover", r"tata", r"e-?type", r"xj|xf|f-type"], "attributes": [r"price", r"engine", r"horsepower", r"dealership"]},
        "software": {"cues": [r"mac\s*os", r"os\s+x", r"operating\s+system", r"apple"], "attributes": [r"release(d)?", r"version"]},
    },
    "bridge": {
        "card_game": {"cues": [r"card", r"contract\s+bridge", r"bidding", r"\btricks?\b", r"\bgame\b"], "attributes": [r"rules", r"players", r"scoring"]},
        "structure": {"cues": [r"river", r"suspension", r"golden\s+gate", r"engineering", r"cross(ing)?", r"span", r"construction"], "attributes": [r"longest", r"tallest", r"built", r"length"]},
        "dental": {"cues": [r"dental", r"teeth", r"tooth", r"dentist"], "attributes": [r"cost", r"procedure"]},
    },
    "springfield": {
        "city": {"cues": [r"illinois", r"missouri", r"massachusetts", r"oregon", r"ohio", r"\bil\b", r"\bmo\b", r"\bma\b"], "attributes": []},
        "fictional": {"cues": [r"simpsons", r"fictional", r"cartoon"], "attributes": []},
    },
    "columbia": {
        "university": {"cues": [r"university", r"college", r"ivy"], "attributes": [r"tuition", r"acceptance\s+rate", r"founded"]},
        "city_or_district": {"cues": [r"south\s+carolina", r"missouri", r"district", r"\bdc\b"], "attributes": []},
        "river": {"cues": [r"river"], "attributes": [r"length", r"flows?"]},
    },
    "alexandria": {
        "egypt": {"cues": [r"egypt", r"library", r"lighthouse", r"ptolem"], "attributes": [r"founded\s+by", r"alexander"]},
        "virginia": {"cues": [r"virginia", r"\bva\b"], "attributes": []},
    },
    "victoria": {
        "queen": {"cues": [r"queen", r"reign", r"victorian\s+era", r"monarch"], "attributes": [r"reign", r"children", r"died"]},
        "australian_state": {"cues": [r"australia", r"melbourne", r"state"], "attributes": [r"premier"]},
        "canadian_city": {"cues": [r"british\s+columbia", r"canada", r"vancouver\s+island"], "attributes": []},
        "lake": {"cues": [r"lake", r"uganda", r"tanzania", r"kenya"], "attributes": [r"area", r"depth"]},
    },
}

# Words that may be capitalised at the start of a question but are not part of a name.
_NON_NAME_WORDS = {
    "what", "who", "whom", "whose", "where", "when", "why", "how", "which", "is", "are", "was", "were", "do",
    "does", "did", "tell", "the", "a", "an", "in", "of", "about", "and", "or", "me", "can", "could", "please",
    "explain", "describe", "define", "give", "show", "name", "list", "i", "its", "it", "to", "from", "for",
}

_CONTEXT_BLOCK_RE = re.compile(r"\[context:\s*([^\]]+)\]", re.IGNORECASE)


@dataclass
class EntityAmbiguityAssessment:
    is_ambiguous: bool
    term: Optional[str] = None
    candidate_senses: List[str] = field(default_factory=list)
    resolved_sense: Optional[str] = None
    reason: str = ""
    evidence: List[str] = field(default_factory=list)


def _adjacent_proper_name(question: str, term: str) -> Optional[str]:
    """Return the neighbouring capitalised token if the term is part of a longer name."""
    tokens = re.findall(r"[A-Za-z][A-Za-z.'\-]*", question)
    for i, tok in enumerate(tokens):
        if tok.lower().strip(".'") != term:
            continue
        for j in (i - 1, i + 1):
            if 0 <= j < len(tokens):
                nb = tokens[j]
                base = nb.lower().strip(".").split("'")[0]  # "Who's" -> "who"
                if (nb[0].isupper() and base not in _NON_NAME_WORDS) or re.fullmatch(r"[A-Z]\.", nb):
                    return nb
    return None


class EntityAmbiguityAssessor:
    """Decides whether a polysemous entity in a question is left unresolved."""

    TERMS = tuple(SENSE_INVENTORY)

    @classmethod
    def assess(cls, question: str) -> EntityAmbiguityAssessment:
        q = question or ""
        q_lower = q.lower()
        for term, senses in SENSE_INVENTORY.items():
            if not re.search(rf"\b{term}\b", q_lower):
                continue

            # 1. Sense cues anywhere in the question (including a [Context: ...] block)
            cued = [s for s, spec in senses.items() if any(re.search(rf"\b{c}", q_lower) for c in spec["cues"])]
            if len(cued) == 1:
                return EntityAmbiguityAssessment(False, term, list(senses), cued[0],
                                                 f"'{term}' is resolved to the {cued[0].replace('_', ' ')} sense by context.",
                                                 [f"sense_cue:{term}:{cued[0]}"])

            # 2. Part of a longer proper name ("Michael Jordan", "Jordan Peterson").
            #    The "[Context: …]" marker itself is not part of the name.
            name = _adjacent_proper_name(_CONTEXT_BLOCK_RE.sub(" ", q), term)
            if name and not cued:
                return EntityAmbiguityAssessment(False, term, list(senses), "named_entity",
                                                 f"'{term}' is part of the proper name '{name} …'.",
                                                 [f"proper_name:{name}"])

            # 3. Attribute that only one sense supports
            attr_senses = None
            for s, spec in senses.items():
                if any(re.search(rf"\b{a}", q_lower) for a in spec["attributes"]):
                    attr_senses = (attr_senses or []) + [s]
            if attr_senses and len(attr_senses) == 1 and not cued:
                return EntityAmbiguityAssessment(False, term, list(senses), attr_senses[0],
                                                 f"Only the {attr_senses[0].replace('_', ' ')} sense of '{term}' has the attribute asked about.",
                                                 [f"attribute:{term}:{attr_senses[0]}"])

            candidates = cued if len(cued) > 1 else (attr_senses if attr_senses else list(senses))
            return EntityAmbiguityAssessment(
                True, term, candidates, None,
                f"'{term}' has multiple common referents ({', '.join(c.replace('_', ' ') for c in candidates)}) "
                "and the question does not select one.",
                [f"polysemic_term:{term}"],
            )
        return EntityAmbiguityAssessment(False, reason="No tracked polysemous entity.")
