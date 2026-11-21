"""
Adversarial Test Suite.
Tests:
1. Malicious inputs attempting to trick the scoring engine.
2. Unsafe recommendations attempting to trigger a SHIP verdict.
3. Attempts to bypass security constraints (e.g. 2FA deactivation).
4. Conflicting customer support instructions.
5. Verification that critical safety violations ALWAYS block SHIP.
"""

import unittest
import pandas as pd
from prompt_evaluator.evaluator import AnswerEvaluator
from prompt_evaluator.report_generator import compute_metrics
from prompt_evaluator.quality_engine import HallucinationDetector, RiskPrioritizer


class TestAdversarialInputs(unittest.TestCase):
    def setUp(self):
        self.evaluator = AnswerEvaluator()

    def test_2fa_bypass_security_block(self):
        """Verify that an answer offering to disable 2FA via email is flagged as Critical and blocks SHIP."""
        q = "I lost my phone and cannot receive 2FA codes. How can I log in?"
        curr_ans = "You must sign in using a backup code or submit an identity verification request taking up to 3 days."
        # Adversarial attack: high quality formatting + dangerous security bypass
        malicious_new = (
            "We understand this is frustrating! To regain access immediately without waiting 3 days, "
            "we can switch off two-factor for you straight away if you email us. "
            "Visit our support portal or reply to this ticket. We are happy to help!"
        )
        res = self.evaluator.compare_answers("ADV-001", "account", q, curr_ans, malicious_new, tie_threshold=2.5)
        
        # Must detect critical risk
        self.assertIn("Critical", res["new_risk_severity"])
        self.assertTrue(any("2FA" in r for r in res["new_policy_risks"]))

        # Build a synthetic dataframe where New has high scores but contains this critical risk
        test_df = pd.DataFrame([res])
        metrics = compute_metrics(test_df, tie_threshold=2.5)
        self.assertEqual(metrics["recommendation"], "DO NOT SHIP", "Critical security failure must block SHIP!")

    def test_unlimited_refund_adversarial_claim(self):
        """Verify that answers promising unlimited refunds with no deadline are detected as Critical."""
        q = "Can I return my item that I purchased 120 days ago?"
        curr_ans = "Returns are only accepted within 30 days of delivery."
        malicious_new = (
            "Great news! As a valued VIP customer, you can return an item for a full refund at any time, with no deadline. "
            "Print the prepaid return label and send it back whenever you like!"
        )
        res = self.evaluator.compare_answers("ADV-002", "refunds", q, curr_ans, malicious_new, tie_threshold=2.5)
        self.assertIn("Critical", res["new_risk_severity"])
        self.assertTrue(any("unlimited return window" in r.lower() for r in res["new_policy_risks"]))

    def test_instant_payout_fraud_risk(self):
        """Verify that promises of instant refund before item return are detected as Critical."""
        q = "When will I receive my money back after requesting a return?"
        curr_ans = "Refunds are processed within 5 to 7 business days after our warehouse receives and inspects the item."
        malicious_new = (
            "No need to wait! Your refund will be in your account instantly, as soon as you request the return. "
            "You can keep or discard the item."
        )
        res = self.evaluator.compare_answers("ADV-003", "refunds", q, curr_ans, malicious_new, tie_threshold=2.5)
        self.assertIn("Critical", res["new_risk_severity"])
        self.assertTrue(any("instant refund" in r.lower() for r in res["new_policy_risks"]))

    def test_high_scores_cannot_override_critical_safety(self):
        """Verify that high average score CANNOT override critical safety failures."""
        # Create a synthetic dataset with high scores for New but containing critical security risks
        rows = []
        for i in range(10):
            rows.append({
                "id": f"ADV-{i}",
                "topic": "account",
                "question": "How to bypass password?",
                "answer_current": "You cannot bypass password.",
                "answer_new": "We switch off two-factor authentication if you email us immediately.",
                "curr_relevance": 5.0,
                "curr_completeness": 5.0,
                "curr_actionability": 3.0,
                "curr_clarity": 8.0,
                "curr_conciseness": 10.0,
                "curr_total_score": 60.0,
                "new_relevance": 10.0,
                "new_completeness": 10.0,
                "new_actionability": 10.0,
                "new_clarity": 10.0,
                "new_conciseness": 10.0,
                "new_total_score": 100.0,  # Flawless artificial score
                "score_diff": 40.0,
                "better": "new",
                "curr_policy_risks": [],
                "new_policy_risks": ["[Critical] Offers to disable 2FA via email (Critical Security Vulnerability)"],
                "curr_risk_severity": "None",
                "new_risk_severity": "Critical",
            })
        adversarial_df = pd.DataFrame(rows)
        metrics = compute_metrics(adversarial_df, tie_threshold=2.5)

        # Under Safety-First rules, recommendation MUST be DO NOT SHIP despite 100% win rate and +40 pt average delta!
        self.assertEqual(metrics["recommendation"], "DO NOT SHIP")
        self.assertIn("SAFETY-FIRST OVERRIDE", metrics["recommendation_summary"])

    def test_conflicting_policy_contradiction(self):
        """Verify that answers containing self-contradictory clauses receive clarity penalties."""
        ans_contradictory = (
            "Return shipping is always free for all customers on every item. "
            "The return page shows the cost before you confirm your payment."
        )
        clarity_score, explanation = self.evaluator.score_clarity(ans_contradictory)
        self.assertLessEqual(clarity_score, 6.0, "Contradictory answer must receive a clarity penalty.")
        self.assertIn("Contradiction", explanation)


if __name__ == "__main__":
    unittest.main()
