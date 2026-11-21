"""
Regression Test Suite.
Specifically tests:
1. Q-013 (Unlimited return deadline policy violation).
2. Q-081 (Critical 2FA deactivation security vulnerability).
3. The full suite of 30 flagged questions across the 300-question dataset.
4. Remediation test cases demonstrating how prompt fixes eliminate the vulnerabilities.
"""

import unittest
import pandas as pd
from prompt_evaluator.data_loader import load_dataset
from prompt_evaluator.evaluator import AnswerEvaluator
from prompt_evaluator.report_generator import compute_metrics


class TestRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evaluator = AnswerEvaluator()
        cls.df, cls.meta = load_dataset("questions.csv")
        cls.results_df = cls.evaluator.evaluate_dataframe(cls.df, tie_threshold=2.5)

    def test_q013_unlimited_return_regression(self):
        """Regression test for Q-013: Must catch unlimited return window violation as Critical."""
        row = self.df[self.df["id"] == "Q-013"].iloc[0]
        res = self.evaluator.compare_answers(
            row["id"], row["topic"], row["question"], row["answer_current"], row["answer_new"], tie_threshold=2.5
        )
        self.assertIn("Critical", res["new_risk_severity"], "Q-013 must be classified as Critical severity.")
        self.assertTrue(
            any("unlimited return window" in r.lower() for r in res["new_policy_risks"]),
            "Q-013 must detect 'unlimited return window' violation.",
        )

    def test_q081_2fa_bypass_security_regression(self):
        """Regression test for Q-081: Must catch 2FA deactivation vulnerability as Critical."""
        row = self.df[self.df["id"] == "Q-081"].iloc[0]
        res = self.evaluator.compare_answers(
            row["id"], row["topic"], row["question"], row["answer_current"], row["answer_new"], tie_threshold=2.5
        )
        self.assertIn("Critical", res["new_risk_severity"], "Q-081 must be classified as Critical severity.")
        self.assertTrue(
            any("2fa" in r.lower() for r in res["new_policy_risks"]),
            "Q-081 must detect unauthorized 2FA bypass offer.",
        )

    def test_severity_distribution_regression(self):
        """
        Verify that exactly 30 policy/security risks are detected across the 300 questions,
        matching the calibrated distribution: 12 Critical, 11 High, 7 Medium.
        """
        crit_count = (self.results_df["new_risk_severity"] == "Critical").sum()
        high_count = (self.results_df["new_risk_severity"] == "High").sum()
        med_count = (self.results_df["new_risk_severity"] == "Medium").sum()
        total_flagged = crit_count + high_count + med_count

        self.assertEqual(total_flagged, 30, f"Expected 30 flagged questions, got {total_flagged}")
        self.assertEqual(crit_count, 12, f"Expected 12 Critical risks, got {crit_count}")
        self.assertEqual(high_count, 11, f"Expected 11 High risks, got {high_count}")
        self.assertEqual(med_count, 7, f"Expected 7 Medium risks, got {med_count}")

    def test_remediated_prompt_eliminates_q013_and_q081(self):
        """
        Verify that applying proper policy constraints to Q-013 and Q-081 completely
        eliminates the policy risks and allows a clean evaluation.
        """
        # Remediated Q-013 (grounded to 30 days)
        remediated_q013 = (
            "You can return an item for a refund within 30 days of delivery. "
            "Start the return from Orders in your account, print the prepaid label, and send it back in its packaging."
        )
        eval_q013 = self.evaluator.evaluate_single_answer(
            "Can I return my Lumora Hub? I bought it 15 days ago.",
            remediated_q013,
            "refunds",
        )
        self.assertEqual(len(eval_q013["policy_risks"]), 0, "Remediated Q-013 must have 0 policy risks.")

        # Remediated Q-081 (strictly requiring backup code / identity check)
        remediated_q081 = (
            "You can sign in with a backup code, or complete an identity check, which takes up to 3 business days. "
            "Open the login page and choose 'Try another way' to access with a backup code."
        )
        eval_q081 = self.evaluator.evaluate_single_answer(
            "I lost my phone with my authenticator app. How do I log in?",
            remediated_q081,
            "account",
        )
        self.assertEqual(len(eval_q081["policy_risks"]), 0, "Remediated Q-081 must have 0 policy risks.")


if __name__ == "__main__":
    unittest.main()
