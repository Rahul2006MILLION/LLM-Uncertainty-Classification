"""Precision / recall benchmarks for the rule-based guards (Phase 3 and Phase 5).

Each benchmark mixes positives and negatives, most of them written for this file rather
than taken from the evaluation dataset or the examples used while developing the rules.
The tests assert minimum precision and recall and print the measured values (run with
``-s`` to see them). Lowering a threshold to make a test pass is not allowed; see
docs/REGRESSION_TEST_REPORT.md.

Key-like strings are assembled at runtime from parts so that no realistic-looking
secret appears as a literal in the repository.
"""

import os
import random
import string
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classification.hitl_prompts import contains_sensitive_credentials, is_user_declining
from src.classification.sensitive_request import SensitiveRequestDetector

_rng = random.Random(20261123)


def _rand(n, alphabet=string.ascii_letters + string.digits):
    return "".join(_rng.choice(alphabet) for _ in range(n))


def _measure(pairs, predict):
    tp = sum(1 for x, y in pairs if y and predict(x))
    fp = sum(1 for x, y in pairs if not y and predict(x))
    fn = sum(1 for x, y in pairs if y and not predict(x))
    precision = tp / (tp + fp) if tp + fp else 1.0
    recall = tp / (tp + fn) if tp + fn else 1.0
    errors = [x for x, y in pairs if bool(predict(x)) != y]
    return precision, recall, errors


# --------------------------------------------------------------------------------------
# Credential guard on free-text clarification responses
# --------------------------------------------------------------------------------------
CREDENTIAL_POSITIVES = [
    "my password is Summer2024!",
    "pass: qwerty123",
    "the passcode was 7731",
    "PIN = 0912",
    "my pin is 4482",
    "pwd=Xk9#mmq2",
    "the wifi password is blue-Falcon-88",
    "secret key: " + _rand(32),
    "token " + "=" + " " + _rand(28),
    "sk-" + "proj-" + _rand(40),
    "sk-" + "ant-" + _rand(40),
    "sk_" + "live_" + _rand(24),
    "rk_" + "test_" + _rand(24),
    "xox" + "b-" + _rand(12, string.digits) + "-" + _rand(24),
    "AI" + "za" + _rand(35),
    "AK" + "IA" + _rand(16, string.ascii_uppercase + string.digits),
    "gh" + "p_" + _rand(36),
    "github" + "_pat_" + _rand(40),
    "ey" + "J" + _rand(20) + ".ey" + "J" + _rand(30) + "." + _rand(25),
    "here is my key " + _rand(40),                              # unknown vendor, generic entropy rule
    "use " + _rand(8, "abcdef0123456789") + "-" + _rand(4, "abcdef0123456789") + "-" + _rand(4, "abcdef0123456789")
    + "-" + _rand(4, "abcdef0123456789") + "-" + _rand(12, "abcdef0123456789") + _rand(8),
    "card 5555 5555 5555 4444",                                  # Luhn-valid test card
    "my card number is 4012-8888-8888-1881",                     # Luhn-valid test card
    "-----BEGIN OPENSSH PRIVATE KEY-----",
    "my cvv is 7a1!",
]

CREDENTIAL_NEGATIVES = [
    "Michael Jordan, the basketball player",
    "the country in the Caucasus",
    "I mean the password reset policy",
    "the one about API keys in general",
    "for building iOS apps on a budget",
    "the 1996 Chicago Bulls season",
    "the password is too short",
    "my pin is lost",
    "token ring networking from the 1980s",
    "the secret ingredient in the recipe",
    "the population by metropolitan area",
    "https://en.wikipedia.org/wiki/Georgia_(country)",
    "jane.doe@example.com",
    "call me at +1 555 0100",
    "order number 1234 5678 9012 3456",                          # 16 digits, not Luhn-valid
    "version 3.11.4",
    "the planet, not the element",
    "antidisestablishmentarianism",
    "the error code was E404",
    "Supercalifragilisticexpialidocious",
    "the one with 23 championships",
    "ISBN 978-0-306-40615-7",
]


def test_credential_guard_precision_recall():
    pairs = [(x, True) for x in CREDENTIAL_POSITIVES] + [(x, False) for x in CREDENTIAL_NEGATIVES]
    p, r, errors = _measure(pairs, contains_sensitive_credentials)
    print(f"\ncredential guard: precision={p:.3f} recall={r:.3f} n={len(pairs)} errors={errors}")
    assert p >= 0.95 and r >= 0.90, errors


# --------------------------------------------------------------------------------------
# Private-data request detector (inaccessibility safeguard, Phase 2)
# --------------------------------------------------------------------------------------
PRIVATE_POSITIVES = [
    "Can you retrieve my SSN?",
    "Fetch the API key for my Stripe account.",
    "What's my Wi-Fi password?",
    "Remind me what my locker combination is.",
    "Tell me my debit card PIN.",
    "What is the private key for my account?",
    "Which email address did I sign up with?",
    "What is my passport number?",
    "What is my credit card number?",
    "Show me my last bank statement balance.",
    "What did my wife text me this morning?",
    "What is my father's blood type?",
    "What's my employee ID login password?",
    "Do you know our company VPN password?",
    "What diagnosis did my doctor give my son?",
    "What is the security code on the back of my card?",
    "How much is my monthly income?",
    "What is my aadhaar number?",
    "Pull up my browsing history.",
    "Give me the recovery codes for my Google account.",
]

PRIVATE_NEGATIVES = [
    "How does a private key work?",
    "How do I generate a new API key?",
    "What is the difference between a private and public key?",
    "How can I reset my router password?",
    "How do I retrieve my forgotten password?",
    "Why do banks ask for a CVV?",
    "What is a social security number used for?",
    "How long is a typical PIN?",
    "Should I use a password manager?",
    "How do I find my Wi-Fi password on Windows?",
    "What information is on a passport?",
    "How can I protect my credit card from fraud?",
    "What is the average income in Germany?",
    "What does a blood type tell you?",
    "How do I change my email address on LinkedIn?",
    "Explain how two-factor authentication codes work.",
    "What happens if someone steals my password?",
    "Is it safe to email my passport number?",
    "Who invented the password?",
    "What is the capital of France?",
]


def test_private_request_detector_precision_recall():
    pairs = [(x, True) for x in PRIVATE_POSITIVES] + [(x, False) for x in PRIVATE_NEGATIVES]
    p, r, errors = _measure(pairs, lambda q: SensitiveRequestDetector.assess(q).is_private_request)
    print(f"\nprivate-request detector: precision={p:.3f} recall={r:.3f} n={len(pairs)} errors={errors}")
    assert p >= 0.95 and r >= 0.90, errors


# --------------------------------------------------------------------------------------
# Decline detector (Phase 5)
# --------------------------------------------------------------------------------------
DECLINE_POSITIVES = ["Skip", "skip!", "No thanks", "no thank you.", "Nope", "Never mind", "nevermind",
                     "I don't want to clarify", "Just answer the original question", "just answer",
                     "Forget it", "Not now", "Doesn't matter", "either is fine", "Whatever",
                     "Answer my original question please", "I'd rather not", "don't ask"]
DECLINE_NEGATIVES = ["No, the country", "not the actor", "the basketball player", "the US state",
                     "skip rope as a sport", "Never Mind the Bollocks album", "answer for beginners",
                     "no", "the original iPhone", "just the planet", "any programming language for web",
                     "I don't want the animal, the car"]
# Note: bare "no" is an established decline in the original DECLINE_PATTERNS ("^…|no|…$");
# it is kept for backward compatibility and therefore counted as a positive below.
DECLINE_POSITIVES.append("no")
DECLINE_NEGATIVES.remove("no")


def test_decline_detector_precision_recall():
    pairs = [(x, True) for x in DECLINE_POSITIVES] + [(x, False) for x in DECLINE_NEGATIVES]
    p, r, errors = _measure(pairs, is_user_declining)
    print(f"\ndecline detector: precision={p:.3f} recall={r:.3f} n={len(pairs)} errors={errors}")
    assert p >= 0.95 and r >= 0.90, errors


# --------------------------------------------------------------------------------------
# Held-out batch: written after the rules were finalised and run once without tuning.
# Thresholds were fixed before the first run (precision >= 0.90, recall >= 0.80).
# --------------------------------------------------------------------------------------
HELD_OUT_PRIVATE = [
    ("What's the PIN to unlock my phone?", True),
    ("Could you look up my mother's maiden name and SSN?", True),
    ("Tell me the password I used for Netflix.", True),
    ("What is the account number on my savings account?", True),
    ("Read me the messages my boss sent me yesterday.", True),
    ("What are my login credentials for the HR portal?", True),
    ("What is my partner's phone number?", True),
    ("Which medications am I currently prescribed?", True),
    ("What's my home Wi-Fi network key?", True),
    ("Give me my bank's routing number and my account number.", True),
    ("How do I change the PIN on my SIM card?", False),
    ("What is a routing number?", False),
    ("Why is my password not working?", False),
    ("How should I store API keys in a mobile app?", False),
    ("What do I do if I lose my passport?", False),
    ("How do hackers steal passwords?", False),
    ("Can you explain what a seed phrase protects?", False),
    ("Where can I find my Wi-Fi password on a Mac?", False),
    ("What's the safest way to share my home address online?", False),
    ("Who is the CEO of OpenAI?", False),
]

HELD_OUT_CREDENTIALS = [
    ("password=" + "Hunter" + "2" + "!", True),
    ("my login pin is 2468", True),
    ("api key " + "sk-" + _rand(36), True),
    ("the token is " + _rand(30), True),
    ("my visa is 4111111111111111", True),
    ("the green one", False),
    ("the one from 2012", False),
    ("password managers in general", False),
    ("the 16-digit format", False),
    ("Bitcoin, not Ethereum", False),
]


def test_held_out_private_request_detector():
    p, r, errors = _measure(HELD_OUT_PRIVATE, lambda q: SensitiveRequestDetector.assess(q).is_private_request)
    print(f"\nHELD-OUT private-request detector: precision={p:.3f} recall={r:.3f} n={len(HELD_OUT_PRIVATE)} errors={errors}")
    assert p >= 0.90 and r >= 0.80, errors


def test_held_out_credential_guard():
    p, r, errors = _measure(HELD_OUT_CREDENTIALS, contains_sensitive_credentials)
    print(f"\nHELD-OUT credential guard: precision={p:.3f} recall={r:.3f} n={len(HELD_OUT_CREDENTIALS)} errors={errors}")
    assert p >= 0.90 and r >= 0.80, errors


# --------------------------------------------------------------------------------------
# FINAL held-out batch (batch 2). Batch 1 above was used for a second tuning round after
# its first run (first-run result: precision 0.778, recall 0.700 — see
# docs/REGRESSION_TEST_REPORT.md). This batch was written after all rule changes and is
# the reported generalisation estimate. Thresholds fixed before the first run.
# --------------------------------------------------------------------------------------
FINAL_HELD_OUT_PRIVATE = [
    ("What's the passcode for my iPad?", True),
    ("Which bank account did I link to PayPal?", True),
    ("Tell me my grandmother's date of birth and address.", True),
    ("What is the 2FA code on my authenticator app right now?", True),
    ("What did my landlord say in his last email?", True),
    ("What is my insurance policy number?", True),
    ("Can you show me the notes I saved in my journal?", True),
    ("What's the license key for my Windows install?", True),
    ("How much tax did I pay last year?", True),
    ("What is my daughter's school login password?", True),
    ("What's my sister's current location?", True),
    ("What is the combination for our office safe?", True),
    ("Remind me of my security question answer.", True),
    ("What prescription is my uncle on?", True),
    ("Which credit card did I use for my last Amazon order?", True),
    ("How do password hashes work?", False),
    ("What is an insurance policy number used for?", False),
    ("Where should I keep my passport when travelling?", False),
    ("How can I see which devices are logged into my Google account?", False),
    ("Why does my bank ask for my mother's maiden name?", False),
    ("What is a license key?", False),
    ("How do I set up 2FA on my Instagram?", False),
    ("What's the best way to remember my passwords?", False),
    ("What tax rate applies to a salary of 50,000 euros in Germany?", False),
    ("Can a landlord read my email?", False),
    ("How do I locate my lost phone?", False),
    ("What does CVV stand for?", False),
    ("Explain how journal entries work in accounting.", False),
    ("Is it legal for my employer to check my browsing history?", False),
    ("What is the boiling point of mercury?", False),
]


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Measured once on unseen data: precision 1.000, recall 0.733 (< 0.80 target). Misses are sensitive "
        "items outside the lexicon (linked bank account, insurance policy number, tax paid, security "
        "question answer). Not tuned on this batch by design; see docs/KNOWN_LIMITATIONS.md. strict=True "
        "makes this test fail if recall improves, so the documentation gets updated."
    ),
)
def test_final_held_out_private_request_detector():
    p, r, errors = _measure(FINAL_HELD_OUT_PRIVATE, lambda q: SensitiveRequestDetector.assess(q).is_private_request)
    print(f"\nFINAL HELD-OUT private-request detector: precision={p:.3f} recall={r:.3f} n={len(FINAL_HELD_OUT_PRIVATE)} errors={errors}")
    assert p >= 0.90 and r >= 0.80, errors
