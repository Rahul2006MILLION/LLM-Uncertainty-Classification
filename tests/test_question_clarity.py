"""Comprehensive regression tests for Question Clarity Assessment layer."""

import pytest
from src.classification.clarity_assessor import QuestionClarityAssessor, ClarityAssessment
from src.classification.router import AdaptiveRouter
from src.classification.fast_path_assessor import FastPathAnswerabilityAssessor
from src.classification.hitl_engine import HITLEngine, ClarificationState, AmbiguityType


class TestQuestionClarityAssessor:
    """Verifies independent Question Clarity Assessment across required 10 target queries."""

    # 1. Missing Referent queries
    def test_missing_referent_which_one_is_better(self):
        q = "Which one is better?"
        a = QuestionClarityAssessor.assess(q)
        assert a.status == "AMBIGUOUS"
        assert a.ambiguity_type == "MISSING_REFERENT"
        assert a.clarification_required is True
        assert a.confidence >= 0.90

    def test_missing_referent_is_it_better(self):
        q = "Is it better?"
        a = QuestionClarityAssessor.assess(q)
        assert a.status == "AMBIGUOUS"
        assert a.ambiguity_type == "MISSING_REFERENT"
        assert a.clarification_required is True

    def test_missing_referent_should_i_choose_it(self):
        q = "Should I choose it?"
        a = QuestionClarityAssessor.assess(q)
        assert a.status == "AMBIGUOUS"
        assert a.ambiguity_type == "MISSING_REFERENT"
        assert a.clarification_required is True

    def test_missing_referent_is_that_good(self):
        q = "Is that good?"
        a = QuestionClarityAssessor.assess(q)
        assert a.status == "AMBIGUOUS"
        assert a.ambiguity_type == "MISSING_REFERENT"
        assert a.clarification_required is True

    # 2. Unambiguous Factual queries (KNOWN)
    def test_unambiguous_capital_of_france(self):
        q = "What is the capital of France?"
        a = QuestionClarityAssessor.assess(q)
        assert a.status == "UNAMBIGUOUS"
        assert a.ambiguity_type is None
        assert a.clarification_required is False
        assert a.confidence >= 0.90

    def test_unambiguous_chemical_symbol_for_gold(self):
        q = "What is the chemical symbol for gold?"
        a = QuestionClarityAssessor.assess(q)
        assert a.status == "UNAMBIGUOUS"
        assert a.ambiguity_type is None
        assert a.clarification_required is False

    def test_unambiguous_force_causes_objects_to_fall(self):
        q = "What force causes objects to fall toward Earth?"
        a = QuestionClarityAssessor.assess(q)
        assert a.status == "UNAMBIGUOUS"
        assert a.ambiguity_type is None
        assert a.clarification_required is False

    # 3. Subjective Superlative queries (AMBIGUOUS)
    def test_subjective_criteria_greatest_scientist(self):
        q = "Who is the greatest scientist of all time?"
        a = QuestionClarityAssessor.assess(q)
        assert a.status == "AMBIGUOUS"
        assert a.ambiguity_type == "SUBJECTIVE_CRITERIA"
        assert a.clarification_required is True

    # 4. Unambiguous Future Unknowable queries (UNKNOWN)
    def test_unambiguous_clarity_with_unknown_answerability(self):
        """Orthogonal separation: specific future demographic query has clear syntax but unknowable answer."""
        q = "What will the exact population of Bengaluru be on January 1, 2040?"
        a = QuestionClarityAssessor.assess(q)
        assert a.status == "UNAMBIGUOUS"
        assert a.ambiguity_type is None
        assert a.clarification_required is False

    # 5. Domain-constrained questions (UNAMBIGUOUS)
    def test_domain_constrained_language_ml(self):
        """Question with specific use-case context ('for machine learning') must not be over-classified as vague."""
        q = "Which programming language is best for machine learning?"
        a = QuestionClarityAssessor.assess(q)
        assert a.status == "UNAMBIGUOUS"
        assert a.clarification_required is False


class TestPipelineClarityRoutingIntegration:
    """Verifies that clarity assessment prevents FAST_PATH routing and enforces HITL consistency."""

    def test_missing_referent_cannot_enter_fast_path(self):
        router = AdaptiveRouter(fast_path_enabled=True)
        missing_referent_queries = [
            ("Which one is better?", "It depends on what you mean."),
            ("Is it better?", "Could you clarify what 'it' refers to?"),
            ("Should I choose it?", "Please specify what you are choosing."),
            ("Is that good?", "That depends on your criteria."),
        ]
        for q, r in missing_referent_queries:
            decision = router.route(q, r)
            assert decision.path == "MULTI_RESPONSE", (
                f"Query '{q}' should have been blocked from FAST_PATH due to missing referent, but got {decision.path}"
            )
            assert "Question clarity assessment flagged ambiguity" in decision.reason

    def test_fast_path_assessor_blocks_missing_referent_from_known(self):
        """Even if direct fact heuristic matches length, fast_path_assessor must not certify missing referents as KNOWN."""
        q = "Which one is better?"
        r = "This question is too vague to answer. Could you please specify which one you are asking about?"
        assessment = FastPathAnswerabilityAssessor.assess(q, r)
        assert assessment.predicted_label != "KNOWN", "Missing referent query must never be certified as KNOWN"
        assert assessment.status == "ASSESSED_FAST_PATH_AMBIGUITY"

    def test_hitl_consistency_on_missing_referent(self):
        """Gate 1 must return CLARIFICATION_REQUIRED for missing referent queries."""
        engine = HITLEngine()
        for q in ["Which one is better?", "Is it better?", "Should I choose it?", "Is that good?"]:
            dec = engine.evaluate_early(q, initial_response="Please clarify.")
            assert dec.decision == ClarificationState.CLARIFICATION_REQUIRED
            assert dec.ambiguity_type == AmbiguityType.MISSING_REFERENT
            assert dec.can_user_resolve is True

    def test_factual_questions_maintain_fast_path(self):
        """Clear factual questions with concise unhedged responses maintain fast path eligibility."""
        router = AdaptiveRouter(fast_path_enabled=True)
        d = router.route("What is the capital of France?", "The capital of France is Paris.")
        assert d.path == "FAST_PATH"

        assessment = FastPathAnswerabilityAssessor.assess(
            "What is the capital of France?", "The capital of France is Paris."
        )
        assert assessment.predicted_label == "KNOWN"
        assert assessment.status == "ASSESSED_FAST_PATH_DIRECT_FACT"
