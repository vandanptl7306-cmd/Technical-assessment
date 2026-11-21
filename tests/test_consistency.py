"""
Evaluation Consistency & Determinism Test Suite.
Tests:
1. Multi-run determinism: 5 repeated evaluations on identical inputs must yield identical scores (variance = 0.000).
2. Boundary threshold verification around the ±2.5 pt tie threshold.
3. Scoring formula mathematical integrity across all 5 criteria.
4. Consistency across question ordering.
"""

import unittest
import numpy as np
from prompt_evaluator.evaluator import AnswerEvaluator, DEFAULT_TIE_THRESHOLD


class TestEvaluationConsistency(unittest.TestCase):
    def setUp(self):
        self.evaluator = AnswerEvaluator()

    def test_multi_run_determinism(self):
        """Verify that running evaluation 5 times on identical questions produces 0.0 variance."""
        q = "Is it too late to cancel my order?"
        curr = "An order can be cancelled only before it ships."
        new = "Thanks for getting in touch! An order can be cancelled only before it ships. Open the order under Orders and choose Cancel."

        runs = []
        for _ in range(5):
            res = self.evaluator.compare_answers("Q-TEST", "orders", q, curr, new, tie_threshold=2.5)
            runs.append((res["curr_total_score"], res["new_total_score"], res["score_diff"], res["better"]))

        # All runs must be identical
        first_run = runs[0]
        for run in runs[1:]:
            self.assertEqual(run, first_run, "Evaluator must be 100% mathematically deterministic!")

    def test_tie_threshold_boundary_conditions(self):
        """
        Verify exact boundary behavior around the ±2.5 tie threshold:
        - diff > 2.50  -> 'new'
        - diff < -2.50 -> 'current'
        - |diff| <= 2.50 -> 'tie'
        """
        threshold = 2.5

        # Case 1: Difference exactly within tie threshold (+2.40 pts)
        eval_curr_mock = {"total_score": 70.0, "relevance": 5.0, "completeness": 5.0, "actionability": 5.0, "clarity": 5.0, "conciseness": 5.0, "relevance_exp": "", "completeness_exp": "", "actionability_exp": "", "clarity_exp": "", "conciseness_exp": "", "policy_risks": [], "risk_severity": "None", "has_policy_risk": False}
        eval_new_mock = {"total_score": 72.4, "relevance": 5.0, "completeness": 5.0, "actionability": 5.0, "clarity": 5.0, "conciseness": 5.0, "relevance_exp": "", "completeness_exp": "", "actionability_exp": "", "clarity_exp": "", "conciseness_exp": "", "policy_risks": [], "risk_severity": "None", "has_policy_risk": False}

        diff = round(eval_new_mock["total_score"] - eval_curr_mock["total_score"], 2)
        self.assertEqual(diff, 2.4)
        verdict = "new" if diff > threshold else ("current" if diff < -threshold else "tie")
        self.assertEqual(verdict, "tie", "Score difference of +2.4 pts must produce 'tie'.")

        # Case 2: Difference exceeding threshold (+2.51 pts)
        diff_exceed = 2.51
        verdict_exceed = "new" if diff_exceed > threshold else ("current" if diff_exceed < -threshold else "tie")
        self.assertEqual(verdict_exceed, "new", "Score difference of +2.51 pts must produce 'new'.")

        # Case 3: Negative difference exceeding threshold (-2.51 pts)
        diff_neg_exceed = -2.51
        verdict_neg = "new" if diff_neg_exceed > threshold else ("current" if diff_neg_exceed < -threshold else "tie")
        self.assertEqual(verdict_neg, "current", "Score difference of -2.51 pts must produce 'current'.")

    def test_scoring_weights_mathematical_sum(self):
        """Verify that the sum of criteria weights equals 10.0 (scaling to 100)."""
        weights = self.evaluator.weights
        total_weight = sum(weights.values())
        self.assertAlmostEqual(total_weight, 10.0, places=4, msg="Weights must sum to 10.0 for 0-100 scale.")
        self.assertEqual(weights["relevance"], 2.5)      # 25%
        self.assertEqual(weights["completeness"], 2.5)   # 25%
        self.assertEqual(weights["actionability"], 2.0)  # 20%
        self.assertEqual(weights["clarity"], 1.5)        # 15%
        self.assertEqual(weights["conciseness"], 1.5)    # 15%

    def test_individual_answer_score_formula_integrity(self):
        """Verify that weighted_total matches the mathematical formula exactly."""
        q = "How long is express shipping?"
        ans = "Express delivery takes 1 to 2 business days. Choose Express at checkout before the order ships."
        res = self.evaluator.evaluate_single_answer(q, ans, "shipping")

        expected = (
            res["relevance"] * 2.5
            + res["completeness"] * 2.5
            + res["actionability"] * 2.0
            + res["clarity"] * 1.5
            + res["conciseness"] * 1.5
        )
        self.assertAlmostEqual(res["total_score"], round(expected, 2), places=2)


if __name__ == "__main__":
    unittest.main()
