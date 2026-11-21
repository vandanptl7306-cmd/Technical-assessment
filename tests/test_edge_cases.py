"""
Edge Case & Stress Test Suite.
Tests:
1. Empty, null, and whitespace-only responses.
2. Extremely long answers (denial-of-service / regex backtracking stress).
3. Missing fields and invalid columns in datasets.
4. Duplicate question IDs.
5. Unexpected data types (numbers, booleans, non-string inputs).
6. Malformed input recovery.
"""

import unittest
import pandas as pd
import io
from prompt_evaluator.evaluator import AnswerEvaluator
from prompt_evaluator.data_loader import load_dataset, validate_dataset, DataValidationError


class TestEdgeCases(unittest.TestCase):
    def setUp(self):
        self.evaluator = AnswerEvaluator()

    def test_empty_and_whitespace_responses(self):
        """Test evaluator handles empty strings and whitespace without throwing exceptions."""
        cases = [
            ("", ""),
            ("   ", "   \t\n"),
            ("Valid question?", ""),
            ("", "Valid answer text."),
        ]
        for q, ans in cases:
            eval_res = self.evaluator.evaluate_single_answer(q, ans, "orders")
            self.assertIsInstance(eval_res["total_score"], float)
            self.assertGreaterEqual(eval_res["total_score"], 0.0)
            self.assertLessEqual(eval_res["total_score"], 100.0)

    def test_extremely_long_response_stress(self):
        """Test resilience against 5,000-word responses without ReDoS or memory blowout."""
        long_answer = "This is a detailed explanation of order return rules. " * 600  # ~6,000 words
        q = "How do I return an item?"
        curr = "Start return from Orders in your account."

        res = self.evaluator.compare_answers("EDGE-LONG", "refunds", q, curr, long_answer, tie_threshold=2.5)
        self.assertIsInstance(res["new_total_score"], float)
        # Should not crash and should penalize conciseness due to low density / excessive length
        self.assertLessEqual(res["new_conciseness"], 7.0)

    def test_missing_required_columns_validation(self):
        """Test that data loader rejects datasets missing required columns."""
        bad_csv = io.StringIO("id,topic,question\nQ-001,orders,Where is my order?")
        with self.assertRaises(DataValidationError) as ctx:
            load_dataset(bad_csv)
        self.assertIn("missing required columns", str(ctx.exception).lower())

    def test_empty_dataset_validation(self):
        """Test that data loader rejects empty datasets."""
        empty_csv = io.StringIO("id,topic,question,answer_current,answer_new\n")
        with self.assertRaises(DataValidationError) as ctx:
            load_dataset(empty_csv)
        self.assertIn("0 rows", str(ctx.exception).lower())

    def test_duplicate_question_ids_warning(self):
        """Test that duplicate IDs are flagged in data validation report."""
        dup_csv = (
            "id,topic,question,answer_current,answer_new\n"
            "Q-001,orders,Cancel order?,Before ships.,Before ships.\n"
            "Q-001,orders,Cancel order?,Before ships.,Before ships.\n"
        )
        df, meta = load_dataset(io.StringIO(dup_csv))
        self.assertEqual(meta["duplicate_ids"], 1)
        self.assertTrue(any("duplicate" in w.lower() for w in meta["warnings"]))

    def test_unexpected_data_types_resilience(self):
        """Test evaluator handles unexpected numeric or boolean inputs gracefully."""
        df_raw = pd.DataFrame([
            {
                "id": 12345,  # int instead of str
                "topic": "orders",
                "question": 9999.0,  # float
                "answer_current": True,  # bool
                "answer_new": None,  # NoneType
            }
        ])
        # Data loader converts all fields to string
        val_meta = validate_dataset(df_raw)
        self.assertTrue(val_meta["is_valid"])

    def test_zero_division_guard_on_empty_metrics(self):
        """Verify report generator handles zero questions gracefully without ZeroDivisionError."""
        from prompt_evaluator.report_generator import compute_metrics
        empty_results = pd.DataFrame(columns=[
            "id", "topic", "better", "score_diff", "curr_total_score", "new_total_score",
            "curr_relevance", "curr_completeness", "curr_actionability", "curr_clarity", "curr_conciseness",
            "new_relevance", "new_completeness", "new_actionability", "new_clarity", "new_conciseness",
            "new_policy_risks"
        ])
        metrics = compute_metrics(empty_results)
        self.assertEqual(metrics["total_questions"], 0)
        self.assertEqual(metrics["curr_avg_score"], 0.0)


if __name__ == "__main__":
    unittest.main()
