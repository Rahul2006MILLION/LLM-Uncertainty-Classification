"""Unit and integration tests for the Reproducible ML Experiment Framework."""

import os
import json
import pytest
import numpy as np

from src.experimentation.feature_groups import (
    FEATURE_NAMES,
    FEATURE_GROUPS,
    EXPERIMENT_CONFIGURATIONS,
    get_features_for_config,
)
from src.experimentation.dataset_loader import (
    ExperimentDatasetLoader,
    DatasetValidationReport,
)
from src.experimentation.data_splitter import (
    LeakageSafeSplitter,
    SplitResult,
)
from src.experimentation.models import (
    ModelFactory,
    MajorityBaselineClassifier,
    SUPPORTED_MODELS,
)
from src.experimentation.metrics import (
    calculate_experiment_metrics,
    MetricsResult,
)
from src.experimentation.tracker import (
    ExperimentTracker,
    ExperimentRecord,
)
from src.experimentation.reporter import (
    ExperimentReporter,
)
from src.experimentation.runner import (
    ExperimentRunner,
)


@pytest.fixture
def mock_dataset_file(tmp_path):
    """Creates a temporary valid feature JSONL dataset for testing."""
    file_path = tmp_path / "mock_features.jsonl"
    classes = ["KNOWN", "AMBIGUOUS", "UNKNOWN"]
    records = []
    for i in range(30):
        c = classes[i % 3]
        rec = {
            "id": f"q_{i:03d}",
            "question": f"Question text number {i} regarding science?",
            "question_group_id": f"grp_{i // 2:02d}",
            "label": c,
            "features": {fn: float(i * 0.1 + j * 0.05) for j, fn in enumerate(FEATURE_NAMES)},
        }
        records.append(rec)

    with open(file_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return str(file_path)


class TestFeatureGroups:
    def test_canonical_feature_count(self):
        assert len(FEATURE_NAMES) == 16
        assert len(set(FEATURE_NAMES)) == 16

    def test_feature_group_coverage(self):
        grouped = set()
        for g_name, f_list in FEATURE_GROUPS.items():
            for f in f_list:
                assert f in FEATURE_NAMES
                grouped.add(f)
        assert grouped == set(FEATURE_NAMES)

    def test_get_features_for_config(self):
        full = get_features_for_config("full_16_features")
        assert len(full) == 16
        single = get_features_for_config("single_response_baseline")
        assert len(single) == 6
        multi = get_features_for_config("multi_response_consistency")
        assert len(multi) == 9

        with pytest.raises(ValueError):
            get_features_for_config("non_existent_config")


class TestDatasetLoader:
    def test_valid_loading_and_matrices(self, mock_dataset_file):
        loader = ExperimentDatasetLoader()
        records, report = loader.load_and_validate(mock_dataset_file)
        assert report.is_valid is True
        assert len(records) == 30
        assert report.total_samples == 30
        assert report.class_distribution == {"KNOWN": 10, "AMBIGUOUS": 10, "UNKNOWN": 10}

        X, y, ids, groups = loader.extract_matrices(records)
        assert X.shape == (30, 16)
        assert y.shape == (30,)
        assert len(ids) == 30
        assert len(groups) == 30

    def test_invalid_labels_caught(self, tmp_path):
        bad_file = tmp_path / "bad_labels.jsonl"
        with open(bad_file, "w") as f:
            f.write(json.dumps({
                "id": "q1",
                "question": "Valid question?",
                "label": "INVALID_LABEL",
                "features": {fn: 1.0 for fn in FEATURE_NAMES},
            }) + "\n")
        loader = ExperimentDatasetLoader()
        records, report = loader.load_and_validate(str(bad_file))
        assert report.is_valid is False
        assert len(report.invalid_labels) == 1

    def test_missing_features_caught(self, tmp_path):
        bad_file = tmp_path / "missing_feat.jsonl"
        with open(bad_file, "w") as f:
            f.write(json.dumps({
                "id": "q1",
                "question": "Valid question?",
                "label": "KNOWN",
                "features": {"semantic_entropy": 0.5},  # Missing other 15 features
            }) + "\n")
        loader = ExperimentDatasetLoader()
        records, report = loader.load_and_validate(str(bad_file))
        assert report.is_valid is False
        assert report.missing_features_count == 15

    def test_duplicate_ids_caught(self, tmp_path):
        dup_file = tmp_path / "duplicate_ids.jsonl"
        with open(dup_file, "w") as f:
            for _ in range(2):
                f.write(json.dumps({
                    "id": "duplicate_id_001",
                    "question": "Valid question?",
                    "label": "KNOWN",
                    "features": {fn: 1.0 for fn in FEATURE_NAMES},
                }) + "\n")
        loader = ExperimentDatasetLoader()
        records, report = loader.load_and_validate(str(dup_file))
        assert report.is_valid is False
        assert "duplicate_id_001" in report.duplicate_ids


class TestLeakageSafeSplitter:
    def test_split_group_disjointness(self, mock_dataset_file):
        loader = ExperimentDatasetLoader()
        records, _ = loader.load_and_validate(mock_dataset_file)
        X, y, ids, groups = loader.extract_matrices(records)

        splitter = LeakageSafeSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42)
        res = splitter.split(records, y, ids, groups=groups)

        assert res.group_disjoint is True
        assert len(set(res.train_ids).intersection(set(res.test_ids))) == 0
        assert len(set(res.train_ids).intersection(set(res.val_ids))) == 0
        assert len(set(res.val_ids).intersection(set(res.test_ids))) == 0

        # Small sample flag
        assert res.is_small_sample is True
        assert len(res.limitations) > 0

    def test_reproducibility(self, mock_dataset_file):
        loader = ExperimentDatasetLoader()
        records, _ = loader.load_and_validate(mock_dataset_file)
        X, y, ids, groups = loader.extract_matrices(records)

        splitter = LeakageSafeSplitter(random_seed=123)
        res1 = splitter.split(records, y, ids, groups=groups)
        res2 = splitter.split(records, y, ids, groups=groups)

        assert res1.train_ids == res2.train_ids
        assert res1.test_ids == res2.test_ids


class TestModelsAndMetrics:
    def test_all_supported_models_instantiation_and_fit(self):
        X = np.random.randn(20, 16).astype(np.float32)
        y = np.random.choice([0, 1, 2], size=20)

        for mname in SUPPORTED_MODELS:
            model = ModelFactory.create_model(mname, random_seed=42)
            model.fit(X, y)
            preds = model.predict(X)
            assert len(preds) == 20
            probs = model.predict_proba(X)
            assert probs.shape == (20, 3)

    def test_metrics_calculation(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 2, 0, 2, 1])
        y_prob = np.eye(3)[y_pred]

        res = calculate_experiment_metrics(y_true, y_pred, y_prob=y_prob)
        assert res.accuracy == 4.0 / 6.0
        assert 0.0 <= res.macro_f1 <= 1.0
        assert res.confusion_matrix[0][0] == 2
        assert res.roc_auc_ovr is not None
        assert res.brier_score is not None


class TestExperimentRunnerAndReporting:
    def test_runner_suite_execution_and_reports(self, mock_dataset_file, tmp_path):
        out_dir = str(tmp_path / "exp_suite_run")
        runner = ExperimentRunner(dataset_path=mock_dataset_file, output_dir=out_dir, random_seed=42)

        # Run 2 models x 2 configs
        models = ["majority_baseline", "logistic_regression"]
        configs = ["single_response_baseline", "full_16_features"]
        records = runner.run_suite(models=models, feature_configs=configs)

        assert len(records) == 4
        assert os.path.exists(os.path.join(out_dir, "split_metadata.json"))
        assert os.path.exists(os.path.join(out_dir, "experiment_suite_report.md"))
        assert os.path.exists(os.path.join(out_dir, "experiment_suite_report.json"))

        # Verify Markdown report content
        with open(os.path.join(out_dir, "experiment_suite_report.md")) as f:
            content = f.read()
        assert "Experiment Suite Report" in content
        assert "Performance Comparison" in content or "Model & Feature Configuration Comparison" in content
        assert "Sample Size Caveat" in content

    def test_scaler_pipeline_isolation(self):
        """Verify StandardScaler in Pipeline is fitted strictly on training data."""
        # Train data with mean 0.0
        X_train = np.random.normal(loc=0.0, scale=1.0, size=(30, 16)).astype(np.float32)
        y_train = np.random.choice([0, 1, 2], size=30)
        # Test data with artificially shifted mean 100.0
        X_test = np.random.normal(loc=100.0, scale=1.0, size=(10, 16)).astype(np.float32)

        pipe = ModelFactory.create_model("logistic_regression", random_seed=42)
        pipe.fit(X_train, y_train)

        scaler = pipe.named_steps["scaler"]
        # Scaler mean must match X_train mean, not X_test
        np.testing.assert_allclose(scaler.mean_, np.mean(X_train, axis=0), atol=1e-5)
        assert np.all(scaler.mean_ < 5.0)  # Cannot have leaked test mean of 100.0

    def test_stratified_class_representation_in_splits(self, mock_dataset_file):
        """Verify train, val, and test partitions all contain all 3 classes."""
        loader = ExperimentDatasetLoader()
        records, _ = loader.load_and_validate(mock_dataset_file)
        X, y, ids, groups = loader.extract_matrices(records)

        splitter = LeakageSafeSplitter(train_ratio=0.70, val_ratio=0.15, test_ratio=0.15, random_seed=42)
        res = splitter.split(records, y, ids, groups=groups)

        for pname, pdist in [("train", res.train_distribution), ("val", res.val_distribution), ("test", res.test_distribution)]:
            for c in ["KNOWN", "AMBIGUOUS", "UNKNOWN"]:
                assert pdist[c] > 0, f"Partition '{pname}' missing class '{c}'"

    def test_ablation_delta_reporting(self, mock_dataset_file, tmp_path):
        """Verify ablation study generates Δ Macro-F1 report."""
        out_dir = str(tmp_path / "exp_ablation_run")
        runner = ExperimentRunner(dataset_path=mock_dataset_file, output_dir=out_dir, random_seed=42)
        records = runner.run_ablation_study(model_name="random_forest")

        assert len(records) == 8
        report_file = os.path.join(out_dir, "experiment_suite_report.md")
        assert os.path.exists(report_file)
        with open(report_file) as f:
            content = f.read()
        assert "Feature Ablation Impact Analysis" in content
        assert "Δ Macro-F1" in content
