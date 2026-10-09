"""Unit tests for semantic final answer selection and ambiguity explanation."""

import unittest
from src.features.semantic_analysis import SemanticCluster
from src.features.answer_extractor import (
    extract_concise_answer,
    select_semantic_final_answer,
)


class TestFinalAnswerSelection(unittest.TestCase):
    def test_extract_concise_answer(self):
        verbose = "The answer is, in short, Paris is the capital of France.\n\nIt is situated on the River Seine."
        concise = extract_concise_answer(verbose)
        self.assertEqual(concise, "Paris is the capital of France.")

    def test_select_semantic_final_answer_dominant_majority(self):
        clusters = [
            SemanticCluster(
                cluster_id=0,
                member_indices=[0, 1, 2, 3, 4, 5, 6, 7],
                size=8,
                proportion=0.8,
                representative_index=0,
                representative_text="The answer is Paris.",
            ),
            SemanticCluster(
                cluster_id=1,
                member_indices=[8, 9],
                size=2,
                proportion=0.2,
                representative_index=8,
                representative_text="Lyon.",
            ),
        ]
        result = select_semantic_final_answer(
            clusters=clusters,
            majority_agreement=0.8,
            second_agreement=0.2,
        )
        self.assertFalse(result["is_split_decision"])
        self.assertEqual(result["final_answer"], "Paris.")
        self.assertIn("80.0%", result["explanation"])

    def test_select_semantic_final_answer_ambiguous_split(self):
        # 5 responses say Atlanta, 5 say Tbilisi
        clusters = [
            SemanticCluster(
                cluster_id=0,
                member_indices=[0, 1, 2, 3, 4],
                size=5,
                proportion=0.5,
                representative_index=0,
                representative_text="Atlanta is the state capital of Georgia.",
            ),
            SemanticCluster(
                cluster_id=1,
                member_indices=[5, 6, 7, 8, 9],
                size=5,
                proportion=0.5,
                representative_index=5,
                representative_text="Tbilisi is the national capital of Georgia.",
            ),
        ]
        result = select_semantic_final_answer(
            clusters=clusters,
            majority_agreement=0.5,
            second_agreement=0.5,
        )
        self.assertTrue(result["is_split_decision"])
        self.assertIn("multiple distinct interpretations", result["final_answer"])
        self.assertIn("Atlanta", result["final_answer"])
        self.assertIn("Tbilisi", result["final_answer"])


if __name__ == "__main__":
    unittest.main()
