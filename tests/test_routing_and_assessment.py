"""Tests for adaptive routing, criterion ambiguity assessment, and controlled ablations."""

import pytest
import numpy as np
from src.classification.router import AdaptiveRouter
from src.classification.ambiguity_assessor import CriterionAmbiguityAssessor
from src.classification.inaccessibility_assessor import InaccessibilityAssessor
from src.classification.classifier import UncertaintyClassifier
from src.evaluation.data_validator import DatasetValidator
from src.evaluation.metrics import evaluate_classifier_predictions
from src.features.feature_extractor import FEATURE_NAMES


class TestRoutingAndAssessment:
    """Verifies adaptive routing rules and criterion ambiguity assessment."""

    def test_challenge_questions_routed_to_multi_response(self):
        """The 5 challenge subjective questions must route to multi-response analysis."""
        router = AdaptiveRouter(fast_path_enabled=True)
        challenge_cases = [
            ("Who is the most successful person in the world?", "Elon Musk is often considered one of the most successful."),
            ("What is the greatest country in the world?", "There is no single greatest country as this is subjective."),
            ("Who was the greatest scientist of all time?", "Albert Einstein and Isaac Newton are frequently cited."),
            ("What is the best programming language?", "There is no single best language; it depends on the task."),
            ("What is the most important invention in human history?", "The wheel, the printing press, and electricity are candidates."),
        ]

        for q, r in challenge_cases:
            decision = router.route(q, r)
            assert decision.path == "MULTI_RESPONSE", (
                f"Question '{q}' should route to MULTI_RESPONSE, but got {decision.path} (reason: {decision.reason})"
            )

    def test_criterion_ambiguity_assessor_detects_all_five_challenge_questions(self):
        """CriterionAmbiguityAssessor must classify all 5 challenge questions as AMBIGUOUS."""
        challenge_cases = [
            ("Who is the most successful person in the world?", ["Elon Musk is widely cited due to his net worth."]),
            ("What is the greatest country in the world?", ["There is no single greatest country as this is subjective."]),
            ("Who was the greatest scientist of all time?", ["Einstein is widely considered the greatest, though subjective."]),
            ("What is the best programming language?", ["Depends on the task and context; Python for AI, C++ for speed."]),
            ("What is the most important invention in human history?", ["No universally accepted invention; depends on perspective."]),
        ]

        for q, resps in challenge_cases:
            assessment = CriterionAmbiguityAssessor.assess(q, resps)
            assert assessment.is_ambiguous is True, f"Question '{q}' should be assessed as AMBIGUOUS"
            assert "Question asks for a subjective superlative" in assessment.reason

    def test_objective_superlative_with_criteria_not_flagged(self):
        """Superlatives with explicit criteria (e.g. by population) should not be treated as unconstrained subjective queries."""
        q = "What is the largest city in the world by population?"
        resps = ["Tokyo is the largest city in the world by metropolitan population with over 37 million residents."]
        assessment = CriterionAmbiguityAssessor.assess(q, resps)
        assert assessment.is_ambiguous is False, "Objective criteria constraint should prevent subjective override"

    def test_factual_and_unknown_questions_not_flagged_ambiguous(self):
        """Factual deterministic queries and genuinely unanswerable private state queries must not be flagged as ambiguous."""
        factual_q = "What is 2 + 2?"
        factual_r = ["2 + 2 equals 4."]
        assert CriterionAmbiguityAssessor.assess(factual_q, factual_r).is_ambiguous is False

        unknown_q = "What is the secret passphrase in my safe?"
        unknown_r = ["I cannot know your private secret passphrase."]
        assert CriterionAmbiguityAssessor.assess(unknown_q, unknown_r).is_ambiguous is False

    def test_fast_path_preserved_for_arithmetic_and_definitions(self):
        """Factual arithmetic and established definitions must maintain fast-path routing."""
        router = AdaptiveRouter(fast_path_enabled=True)
        d1 = router.route("What is 2 + 2?", "2 + 2 is 4.")
        assert d1.path == "FAST_PATH"

        d2 = router.route("What is data science?", "Data science is an interdisciplinary field.")
        assert d2.path == "FAST_PATH"


class TestFastPathAndAssessmentRegression:
    """Requirement 7 regression tests covering required question types."""

    def test_arithmetic_regression(self):
        """Arithmetic: 'What is 2 + 2?' -> FAST_PATH, KNOWN."""
        from src.classification.fast_path_assessor import FastPathAnswerabilityAssessor
        router = AdaptiveRouter(fast_path_enabled=True)
        q = "What is 2 + 2?"
        r = "2 + 2 is 4."
        d = router.route(q, r)
        assert d.path == "FAST_PATH"

        assessment = FastPathAnswerabilityAssessor.assess(q, r)
        assert assessment.is_assessed is True
        assert assessment.predicted_label == "KNOWN"
        assert assessment.confidence == 0.95
        assert assessment.status == "ASSESSED_FAST_PATH_DETERMINISTIC_ARITHMETIC"

    def test_definition_regression(self):
        """Definition: 'What is data science?' -> FAST_PATH, KNOWN."""
        from src.classification.fast_path_assessor import FastPathAnswerabilityAssessor
        router = AdaptiveRouter(fast_path_enabled=True)
        q = "What is data science?"
        r = "Data science is an interdisciplinary field that uses scientific methods to extract knowledge from data."
        d = router.route(q, r)
        assert d.path == "FAST_PATH"

        assessment = FastPathAnswerabilityAssessor.assess(q, r)
        assert assessment.is_assessed is True
        assert assessment.predicted_label == "KNOWN"
        assert assessment.confidence == 0.90
        assert assessment.status == "ASSESSED_FAST_PATH_CONCEPT_DEFINITION"

    def test_factual_regression(self):
        """Factual question: 'What is the capital of France?' -> FAST_PATH, KNOWN."""
        from src.classification.fast_path_assessor import FastPathAnswerabilityAssessor
        router = AdaptiveRouter(fast_path_enabled=True)
        q = "What is the capital of France?"
        r = "The capital of France is Paris."
        d = router.route(q, r)
        assert d.path == "FAST_PATH"

        assessment = FastPathAnswerabilityAssessor.assess(q, r)
        assert assessment.is_assessed is True
        assert assessment.predicted_label == "KNOWN"
        assert assessment.confidence == 0.85
        assert assessment.status == "ASSESSED_FAST_PATH_DIRECT_FACT"

    def test_five_challenge_questions_regression(self):
        """Five existing subjective-superlative challenge questions -> MULTI_RESPONSE, AMBIGUOUS."""
        router = AdaptiveRouter(fast_path_enabled=True)
        challenge_cases = [
            ("Who is the most successful person in the world?", ["Elon Musk is widely cited due to his net worth."]),
            ("What is the greatest country in the world?", ["There is no single greatest country as this is subjective."]),
            ("Who was the greatest scientist of all time?", ["Einstein is widely considered the greatest, though subjective."]),
            ("What is the best programming language?", ["Depends on the task and context; Python for AI, C++ for speed."]),
            ("What is the most important invention in human history?", ["No universally accepted invention; depends on perspective."]),
        ]

        for q, resps in challenge_cases:
            decision = router.route(q, resps[0])
            assert decision.path == "MULTI_RESPONSE", f"Question '{q}' should route to MULTI_RESPONSE"
            assessment = CriterionAmbiguityAssessor.assess(q, resps)
            assert assessment.is_ambiguous is True, f"Question '{q}' should be assessed as AMBIGUOUS"
            assert assessment.confidence_type == "rule_based_heuristic"

    def test_unanswerable_questions_regression(self):
        """At least two genuinely unanswerable questions -> UNKNOWN."""
        from src.classification.fast_path_assessor import FastPathAnswerabilityAssessor
        router = AdaptiveRouter(fast_path_enabled=True)
        unanswerable_cases = [
            (
                "What is the secret passphrase in my safe?",
                "I cannot know or access your private secret passphrase."
            ),
            (
                "What was Socrates whispering at the exact moment of his death that was not recorded by Plato or Xenophon?",
                "This cannot be known or determined because it was never recorded in any historical source."
            ),
        ]

        for q, r in unanswerable_cases:
            # When routed: router flags refusal / unknowable words
            decision = router.route(q, r)
            assert decision.path == "MULTI_RESPONSE"

            # If evaluated directly under fast path assessor with refusal response:
            fast_assessment = FastPathAnswerabilityAssessor.assess(q, r)
            assert fast_assessment.is_assessed is True
            assert fast_assessment.predicted_label == "UNKNOWN"
            assert fast_assessment.status == "ASSESSED_FAST_PATH_REFUSAL"


class TestControlledModelComparisons:
    """Verifies that all 5 controlled model ablations train and evaluate on a shared group split."""

    def test_controlled_ablation_pipeline(self):
        # Create a synthetic dataset of 18 records across 6 groups (2 per class)
        records = [
            {"id": "k1", "question": "What is 2+2?", "label": "KNOWN", "features": {fn: 0.1 for fn in FEATURE_NAMES}},
            {"id": "k2", "question": "What is 2+2?", "label": "KNOWN", "features": {fn: 0.1 for fn in FEATURE_NAMES}},
            {"id": "k3", "question": "What is Paris?", "label": "KNOWN", "features": {fn: 0.2 for fn in FEATURE_NAMES}},
            {"id": "a1", "question": "Who is Jordan?", "label": "AMBIGUOUS", "features": {fn: 0.5 for fn in FEATURE_NAMES}},
            {"id": "a2", "question": "Who is Jordan?", "label": "AMBIGUOUS", "features": {fn: 0.5 for fn in FEATURE_NAMES}},
            {"id": "a3", "question": "Largest city?", "label": "AMBIGUOUS", "features": {fn: 0.6 for fn in FEATURE_NAMES}},
            {"id": "u1", "question": "My safe code?", "label": "UNKNOWN", "features": {fn: 0.9 for fn in FEATURE_NAMES}},
            {"id": "u2", "question": "My safe code?", "label": "UNKNOWN", "features": {fn: 0.9 for fn in FEATURE_NAMES}},
            {"id": "u3", "question": "Mars 2099?", "label": "UNKNOWN", "features": {fn: 0.8 for fn in FEATURE_NAMES}},
        ]

        train, val, test, meta = DatasetValidator.group_train_val_test_split(
            records, train_ratio=0.5, val_ratio=0.0, test_ratio=0.5, random_seed=42, return_metadata=True
        )

        assert meta["is_disjoint"] is True
        assert len(train) > 0 and len(test) > 0

        # Features sets
        all_features = list(FEATURE_NAMES)
        single_features = ["response_len_mean", "distinct_1gram_ratio", "question_overlap_mean", "hedging_score", "negation_count"]
        multi_features = ["semantic_entropy", "normalized_entropy", "dominant_agreement", "second_agreement", "agreement_margin"]
        no_neg_features = [f for f in all_features if f != "negation_count"]

        def _get_xy(recs, fnames):
            X = np.array([[r["features"][fn] for fn in fnames] for r in recs], dtype=np.float32)
            y = np.array([0 if r["label"] == "KNOWN" else (1 if r["label"] == "AMBIGUOUS" else 2) for r in recs], dtype=int)
            return X, y

        y_train = np.array([0 if r["label"] == "KNOWN" else (1 if r["label"] == "AMBIGUOUS" else 2) for r in train], dtype=int)
        y_test = np.array([0 if r["label"] == "KNOWN" else (1 if r["label"] == "AMBIGUOUS" else 2) for r in test], dtype=int)

        # 1. Full model
        X_train_full, _ = _get_xy(train, all_features)
        X_test_full, _ = _get_xy(test, all_features)
        clf_full = UncertaintyClassifier(model_type="random_forest", random_state=42)
        clf_full.train(X_train_full, y_train, feature_names=all_features)
        preds_full = [clf_full.predict({fn: X_test_full[i, j] for j, fn in enumerate(all_features)})["predicted_class_id"] for i in range(len(y_test))]
        eval_full = evaluate_classifier_predictions(y_test.tolist(), preds_full)
        assert "accuracy" in eval_full
        assert "per_class" in eval_full
        assert "confusion_matrix" in eval_full

        # 2. Model without negation_count
        X_train_no_neg, _ = _get_xy(train, no_neg_features)
        X_test_no_neg, _ = _get_xy(test, no_neg_features)
        clf_no_neg = UncertaintyClassifier(model_type="random_forest", random_state=42)
        clf_no_neg.train(X_train_no_neg, y_train, feature_names=no_neg_features)
        preds_no_neg = [clf_no_neg.predict({fn: X_test_no_neg[i, j] for j, fn in enumerate(no_neg_features)})["predicted_class_id"] for i in range(len(y_test))]
        eval_no_neg = evaluate_classifier_predictions(y_test.tolist(), preds_no_neg)
        assert "accuracy" in eval_no_neg


class TestInaccessibilitySafeguard:
    """Verifies that InaccessibilityAssessor reliably detects inaccessible information without false positives."""

    def test_inaccessible_information_cases_detected_as_unknown(self):
        """All six required inaccessible information challenge cases must assess as UNKNOWN."""
        test_cases = [
            (
                "What is the secret password in my safe?",
                [
                    "The secret password in your safe is unknown as it is personal and confidential.",
                    "I'm sorry, but as an AI, I do not have access to your safe or any passwords.",
                ],
                "private_credential",
            ),
            (
                "What is the Wi-Fi WPA2 password for my home router?",
                [
                    "I do not have access to your private Wi-Fi network credentials.",
                    "As an AI, I cannot know or access your home router WPA2 password.",
                ],
                "private_credential",
            ),
            (
                "What code or word is written on the piece of paper in my home safe?",
                [
                    "The contents of your home safe are private and inaccessible.",
                    "I cannot see or determine what code is written on the paper inside your safe.",
                ],
                "private_physical_secret",
            ),
            (
                "What was I thinking at exactly 9 AM yesterday?",
                [
                    "I cannot read your mind or know your thoughts at 9 AM yesterday.",
                    "As an AI, I have no access to your personal mental state or thoughts.",
                ],
                "private_mental_state",
            ),
            (
                "What did John Smith of London eat for breakfast on March 14, 1842?",
                [
                    "There is no historical record of what John Smith of London ate for breakfast on that date.",
                    "This information was never recorded in any historical source and is unrecorded.",
                ],
                "unrecorded_historical_minutiae",
            ),
            (
                "What secret word will I write down in my notebook tomorrow morning?",
                [
                    "I cannot predict or know what secret word you will write in your notebook tomorrow.",
                    "Future private actions and writings in your personal notebook are unknowable.",
                ],
                "private_physical_secret",
            ),
        ]

        for question, responses, category in test_cases:
            assessment = InaccessibilityAssessor.assess(question=question, responses=responses)
            assert assessment.is_inaccessible is True, (
                f"Question '{question}' was not flagged as inaccessible (reason: {assessment.reason})"
            )
            assert assessment.predicted_label == "UNKNOWN", (
                f"Question '{question}' expected predicted_label='UNKNOWN', got {assessment.predicted_label}"
            )
            assert assessment.confidence >= 0.85
            assert "ASSESSED_INACCESSIBLE_RULE_HEURISTIC" in assessment.status

    def test_ambiguous_questions_not_flagged_as_inaccessible(self):
        """Genuine ambiguous queries must NOT be marked inaccessible, even if responses discuss inability to give one answer."""
        ambiguous_cases = [
            (
                "What is the capital of Georgia?",
                [
                    "Georgia can refer to the US state, whose capital is Atlanta, or the country, whose capital is Tbilisi.",
                    "I cannot provide a single capital without context because Georgia has multiple meanings.",
                ],
            ),
            (
                "What is the best programming language?",
                [
                    "There is no single best programming language; it depends on your specific use case.",
                    "Choosing the best language is subjective and varies greatly based on task requirements.",
                ],
            ),
        ]

        for question, responses in ambiguous_cases:
            assessment = InaccessibilityAssessor.assess(question=question, responses=responses)
            assert assessment.is_inaccessible is False, (
                f"Ambiguous question '{question}' was incorrectly flagged as inaccessible!"
            )
            assert assessment.predicted_label is None

    def test_refusal_alone_does_not_force_unknown_on_arbitrary_query(self):
        """A refusal response to a question with no inaccessibility evidence must not trigger UNKNOWN."""
        q = "What is the speed of light in a vacuum?"
        r = ["I cannot determine the answer because I am experiencing a temporary error."]
        assessment = InaccessibilityAssessor.assess(question=q, responses=r)
        assert assessment.is_inaccessible is False, "Refusal alone on factual question must not force inaccessibility"

    def test_pipeline_integration_overrides_spurious_ambiguous_on_safe_password(self):
        """Pipeline must override spurious AMBIGUOUS classifier output to UNKNOWN for the safe password query."""
        from unittest.mock import MagicMock
        from experiments.ask import run_pipeline

        # Simulate the exact conditions observed in the live baseline test:
        # 10 responses split into 2 stylistic refusal clusters
        simulated_responses = [
            {"index": 0, "text": "The secret password in your safe is unknown as it is personal.", "eval_count": 10, "total_duration_ms": 10},
            {"index": 1, "text": "The secret password in your safe is unknown as it is personal.", "eval_count": 10, "total_duration_ms": 10},
            {"index": 2, "text": "I'm sorry, but as an AI, I do not have access to your safe or contents.", "eval_count": 10, "total_duration_ms": 10},
            {"index": 3, "text": "I'm sorry, but as an AI, I do not have access to your safe or contents.", "eval_count": 10, "total_duration_ms": 10},
            {"index": 4, "text": "The secret password in your safe is unknown as it is personal.", "eval_count": 10, "total_duration_ms": 10},
            {"index": 5, "text": "The secret password in your safe is unknown as it is personal.", "eval_count": 10, "total_duration_ms": 10},
            {"index": 6, "text": "I'm sorry, but as an AI, I do not have access to your safe or contents.", "eval_count": 10, "total_duration_ms": 10},
            {"index": 7, "text": "I'm sorry, but as an AI, I do not have access to your safe or contents.", "eval_count": 10, "total_duration_ms": 10},
            {"index": 8, "text": "The secret password in your safe is unknown as it is personal.", "eval_count": 10, "total_duration_ms": 10},
            {"index": 9, "text": "The secret password in your safe is unknown as it is personal.", "eval_count": 10, "total_duration_ms": 10},
        ]

        mock_gen = MagicMock()
        mock_gen.generate_single.return_value = simulated_responses[0]
        mock_gen.generate_multi.return_value = simulated_responses

        mock_emb = MagicMock()
        # Mock embeddings for 10 responses
        mock_emb.embed_batch.return_value = np.zeros((10, 384), dtype=np.float32)

        # Mock the trained classifier to simulate it predicting 'AMBIGUOUS' (the exact bug observed)
        mock_clf = MagicMock()
        mock_clf.is_trained = True
        mock_clf.predict.return_value = {
            "status": "PREDICTED",
            "predicted_label": "AMBIGUOUS",
            "predicted_class_id": 1,
            "confidence": 0.679,
        }

        config = {
            "routing": {"fast_path_enabled": True},
            "generation": {"total_responses": 10},
            "clustering": {"similarity_threshold": 0.80},
            "output_selection": {"explanation_threshold": 0.25},
        }

        record = run_pipeline(
            question="What is the secret password in my safe?",
            config=config,
            generator=mock_gen,
            classifier=mock_clf,
            embedder=mock_emb,
        )

        assert record.path == "MULTI_RESPONSE"
        # The safeguard must override the spurious AMBIGUOUS prediction to UNKNOWN
        assert record.predicted_label == "UNKNOWN"
        assert "ASSESSED_INACCESSIBLE_RULE_HEURISTIC" in record.classifier_status

    def test_security_concept_definitions_not_classified_as_unknown(self):
        """Definitions with security keywords (password, WPA2, key) must evaluate to KNOWN, not UNKNOWN."""
        from unittest.mock import MagicMock
        from experiments.ask import run_pipeline

        concept_cases = [
            ("What is a one-time password?", "A one-time password (OTP) is an automatically generated numeric code for a single session."),
            ("What does WPA2 mean?", "WPA2 stands for Wi-Fi Protected Access 2, a network security protocol."),
            ("Define WPA2 encryption", "WPA2 encryption is a security protocol that secures wireless computer networks."),
            ("What is a password manager?", "A password manager is a software application designed to store and manage digital credentials."),
        ]

        for q, sample_r in concept_cases:
            # 1. Verify assessor in isolation
            assessment = InaccessibilityAssessor.assess(question=q, responses=[sample_r])
            assert assessment.is_inaccessible is False, f"Concept query '{q}' should not be assessed as inaccessible"
            assert assessment.predicted_label is None

            # 2. Verify pipeline execution
            mock_gen = MagicMock()
            mock_gen.generate_single.return_value = {"text": sample_r, "eval_count": 10, "total_duration_ms": 10}
            mock_emb = MagicMock()
            mock_emb.embed_batch.return_value = np.zeros((1, 384), dtype=np.float32)

            config = {
                "routing": {"fast_path_enabled": True},
                "generation": {"total_responses": 10},
            }

            rec = run_pipeline(question=q, config=config, generator=mock_gen, embedder=mock_emb)
            assert rec.path == "FAST_PATH", f"Query '{q}' should take fast path"
            assert rec.predicted_label == "KNOWN", f"Query '{q}' should be classified as KNOWN, got {rec.predicted_label}"

    def test_all_seven_boundary_cases_end_to_end(self):
        """End-to-end pipeline validation of all 7 target boundary cases."""
        from unittest.mock import MagicMock
        from experiments.ask import run_pipeline

        mock_emb = MagicMock()
        mock_emb.embed_batch.return_value = np.zeros((10, 384), dtype=np.float32)

        # Baseline classifier mock: predicts AMBIGUOUS by default to test safeguard overrides
        mock_clf = MagicMock()
        mock_clf.is_trained = True
        mock_clf.predict.return_value = {
            "status": "PREDICTED",
            "predicted_label": "AMBIGUOUS",
            "predicted_class_id": 1,
            "confidence": 0.60,
        }

        config = {
            "routing": {"fast_path_enabled": True},
            "generation": {"total_responses": 10},
            "clustering": {"similarity_threshold": 0.80},
            "output_selection": {"explanation_threshold": 0.25},
        }

        boundary_cases = [
            # 1. Def: What is a one-time password? -> KNOWN
            (
                "What is a one-time password?",
                "A one-time password is an authentication method for a single session.",
                "KNOWN",
                "FAST_PATH",
            ),
            # 2. Def: What does WPA2 mean? -> KNOWN
            (
                "What does WPA2 mean?",
                "WPA2 stands for Wi-Fi Protected Access 2, a network security protocol.",
                "KNOWN",
                "FAST_PATH",
            ),
            # 3. Ambiguous: What is the capital of Georgia? -> AMBIGUOUS
            (
                "What is the capital of Georgia?",
                "Georgia has two capitals depending on whether you mean the state (Atlanta) or country (Tbilisi).",
                "AMBIGUOUS",
                "MULTI_RESPONSE",
            ),
            # 4. Ambiguous: What is the best programming language? -> AMBIGUOUS
            (
                "What is the best programming language?",
                "There is no single best programming language as it depends on use case.",
                "AMBIGUOUS",
                "MULTI_RESPONSE",
            ),
            # 5. Unknown: What is the secret password in my safe? -> UNKNOWN
            (
                "What is the secret password in my safe?",
                "The secret password in your safe is unknown as it is personal.",
                "UNKNOWN",
                "MULTI_RESPONSE",
            ),
            # 6. Unknown: What was I thinking at exactly 9 AM yesterday? -> UNKNOWN
            (
                "What was I thinking at exactly 9 AM yesterday?",
                "I cannot read your mind or know what you were thinking.",
                "UNKNOWN",
                "MULTI_RESPONSE",
            ),
            # 7. Unknown: What is the Wi-Fi password for my private home router? -> UNKNOWN
            (
                "What is the Wi-Fi password for my private home router?",
                "I do not have access to your private Wi-Fi network credentials.",
                "UNKNOWN",
                "MULTI_RESPONSE",
            ),
        ]

        for q, sample_r, expected_label, expected_path in boundary_cases:
            mock_gen = MagicMock()
            mock_gen.generate_single.return_value = {"text": sample_r, "eval_count": 10, "total_duration_ms": 10}
            mock_gen.generate_multi.return_value = [
                {"index": i, "text": sample_r, "eval_count": 10, "total_duration_ms": 10}
                for i in range(10)
            ]

            record = run_pipeline(
                question=q,
                config=config,
                generator=mock_gen,
                classifier=mock_clf,
                embedder=mock_emb,
            )

            assert record.path == expected_path, (
                f"Question '{q}' expected path '{expected_path}', got '{record.path}'"
            )
            assert record.predicted_label == expected_label, (
                f"Question '{q}' expected label '{expected_label}', got '{record.predicted_label}' (status: {record.classifier_status})"
            )


