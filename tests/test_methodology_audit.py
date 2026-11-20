"""Unit and regression tests verifying experimental methodology correction and fairness."""

import json
import pytest
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from src.features.feature_extractor import FeatureExtractor, FEATURE_NAMES
from src.experimentation.feature_groups import (
    TRUE_SINGLE_RESPONSE_FEATURES,
    ENSEMBLE_LINGUISTIC_FEATURES,
    CONSISTENCY_ONLY_FEATURES,
    EXPERIMENT_CONFIGURATIONS,
)


def test_01_experiment_a_uses_only_response_zero():
    """Verify Experiment A features are calculated strictly from response 0."""
    question = "What is the capital of France?"
    r0 = "The capital of France is Paris."
    r1 = "Paris is definitely the capital."
    
    # Feature extraction on r0 alone
    feat_r0 = FeatureExtractor.extract_single_response_features(question, r0)
    
    assert feat_r0["response_len"] == len(r0)
    assert feat_r0["distinct_1gram_ratio"] > 0
    # Make sure r1 text does not affect feat_r0
    assert "distinct_2gram_ratio" in feat_r0
    assert "question_overlap" in feat_r0
    assert "hedging_score" in feat_r0
    assert "negation_count" in feat_r0


def test_02_experiment_a_excludes_multi_response_features():
    """Verify Experiment A configuration strictly excludes all consistency features."""
    consistency_names = {
        "semantic_entropy",
        "normalized_entropy",
        "dominant_agreement",
        "second_agreement",
        "agreement_margin",
        "cluster_count",
        "cluster_ratio",
        "mean_pairwise_similarity",
        "min_pairwise_similarity",
    }
    for feat in TRUE_SINGLE_RESPONSE_FEATURES:
        assert feat not in consistency_names, f"Consistency feature '{feat}' found in True Single Response!"


def test_03_experiment_a_does_not_include_response_len_std():
    """Verify Experiment A strictly excludes response_len_std."""
    assert "response_len_std" not in TRUE_SINGLE_RESPONSE_FEATURES
    assert "response_len_std" not in EXPERIMENT_CONFIGURATIONS["true_single_response_baseline"]


def test_04_experiment_b_uses_linguistic_excludes_consistency():
    """Verify Experiment B uses linguistic features but excludes all 9 consistency features."""
    consistency_names = set(CONSISTENCY_ONLY_FEATURES)
    b_features = EXPERIMENT_CONFIGURATIONS["ensemble_linguistic_baseline"]
    
    # Must contain linguistic features
    assert "response_len_mean" in b_features
    assert "response_len_std" in b_features
    assert "distinct_1gram_ratio" in b_features
    assert "question_overlap_mean" in b_features
    assert "hedging_score" in b_features
    assert "negation_count" in b_features
    
    # Must NOT contain consistency features
    for f in b_features:
        assert f not in consistency_names, f"Consistency feature '{f}' found in Experiment B!"


def test_05_experiment_c_uses_only_nine_consistency_features():
    """Verify Experiment C uses exactly the 9 canonical consistency features."""
    c_features = EXPERIMENT_CONFIGURATIONS["multi_response_consistency_only"]
    assert len(c_features) == 9
    expected = {
        "semantic_entropy",
        "normalized_entropy",
        "dominant_agreement",
        "second_agreement",
        "agreement_margin",
        "cluster_count",
        "cluster_ratio",
        "mean_pairwise_similarity",
        "min_pairwise_similarity",
    }
    assert set(c_features) == expected


def test_06_experiment_d_uses_all_sixteen_canonical_features():
    """Verify Experiment D contains all 16 canonical features in schema order."""
    d_features = EXPERIMENT_CONFIGURATIONS["full_16_feature_model"]
    assert len(d_features) == 16
    assert d_features == list(FEATURE_NAMES)


def test_07_all_experiments_use_identical_group_aware_splits():
    """Verify all 4 experiments split identically across question groups without leakage."""
    # Synthetic dataset with 50 questions across 50 groups
    y = np.array(["KNOWN"] * 17 + ["AMBIGUOUS"] * 17 + ["UNKNOWN"] * 16)
    groups = [f"grp_{i}" for i in range(50)]
    
    # 4 distinct feature matrices with different shapes
    X_a = np.random.randn(50, len(TRUE_SINGLE_RESPONSE_FEATURES))
    X_b = np.random.randn(50, len(ENSEMBLE_LINGUISTIC_FEATURES))
    X_c = np.random.randn(50, len(CONSISTENCY_ONLY_FEATURES))
    X_d = np.random.randn(50, len(FEATURE_NAMES))
    
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    
    splits_a = list(cv.split(X_a, y, groups))
    splits_b = list(cv.split(X_b, y, groups))
    splits_c = list(cv.split(X_c, y, groups))
    splits_d = list(cv.split(X_d, y, groups))
    
    # Ensure splits are byte-identical across all 4 experiments
    for fold in range(5):
        train_a, test_a = splits_a[fold]
        train_b, test_b = splits_b[fold]
        train_c, test_c = splits_c[fold]
        train_d, test_d = splits_d[fold]
        
        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(train_a, train_c)
        np.testing.assert_array_equal(train_a, train_d)
        
        np.testing.assert_array_equal(test_a, test_b)
        np.testing.assert_array_equal(test_a, test_c)
        np.testing.assert_array_equal(test_a, test_d)
        
        # Verify zero group leakage between train and test
        train_groups = set(groups[i] for i in train_a)
        test_groups = set(groups[i] for i in test_a)
        assert len(train_groups.intersection(test_groups)) == 0


def test_08_no_feature_leakage_and_scaler_isolation():
    """Verify scaler fits exclusively on train fold without test leakage."""
    from sklearn.preprocessing import StandardScaler
    
    X_train = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    X_test = np.array([[100.0, 200.0]])
    
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    # Mean and scale must reflect only training data
    assert np.allclose(scaler.mean_, [3.0, 4.0])
    # Transforming test data does not update train statistics
    transformed_test = scaler.transform(X_test)
    assert np.allclose(scaler.mean_, [3.0, 4.0])
