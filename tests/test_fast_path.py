"""Unit tests for fast-path and uncertainty-trigger decisions."""

import unittest
from unittest.mock import patch
from experiments.ask import assess_uncertainty


class TestFastPathDecisions(unittest.TestCase):
    def test_assess_uncertainty_fast_path(self):
        fake_assessor_response = "FAST_PATH\nDirect and accurate definition with no ambiguity."
        with patch("experiments.ask.generate_response", return_value=fake_assessor_response):
            decision, reason = assess_uncertainty(
                question="What is 2 + 2?",
                initial_answer="2 + 2 is 4.",
            )
            self.assertEqual(decision, "FAST_PATH")
            self.assertIn("Direct and accurate", reason)

    def test_assess_uncertainty_analyze_10(self):
        fake_assessor_response = "ANALYZE_10\nAmbiguous between Georgia state and country."
        with patch("experiments.ask.generate_response", return_value=fake_assessor_response):
            decision, reason = assess_uncertainty(
                question="What is the capital of Georgia?",
                initial_answer="The capital is Atlanta.",
            )
            self.assertEqual(decision, "ANALYZE_10")
            self.assertIn("Ambiguous", reason)

    def test_assess_uncertainty_malformed_response_fallback(self):
        # If assessor returns an unparseable response, fail safely to ANALYZE_10
        fake_assessor_response = "I am not sure what format to use."
        with patch("experiments.ask.generate_response", return_value=fake_assessor_response):
            decision, reason = assess_uncertainty(
                question="Some question",
                initial_answer="Some answer",
            )
            self.assertEqual(decision, "ANALYZE_10")
            self.assertIn("Ambiguity or unverified assessment", reason)


if __name__ == "__main__":
    unittest.main()
