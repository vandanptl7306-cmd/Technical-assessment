"""
Unified Test Runner for Prompt Evaluator.
Executes all test suites:
- Baseline Test Suite (run_tests.py)
- Adversarial Test Suite (tests/test_adversarial.py)
- Prompt Injection Test Suite (tests/test_prompt_injection.py)
- Edge Cases & Stress Test Suite (tests/test_edge_cases.py)
- Regression Test Suite (tests/test_regression.py, specifically Q-013 & Q-081)
- Consistency & Determinism Test Suite (tests/test_consistency.py)
- Security & Artifact Integrity Test Suite (tests/test_security.py)

Exits with code 0 on success, code 1 on failure.
"""

import sys
import unittest
import time

def run_all_suites():
    print("=" * 70)
    print("🚀 RUNNING FULL AI EVALUATION TEST SUITE")
    print("=" * 70)

    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Discover and add test modules from tests/
    suite.addTests(loader.discover("tests", pattern="test_*.py"))

    start_time = time.perf_counter()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    elapsed = time.perf_counter() - start_time

    print("\n" + "=" * 70)
    print(f"📊 TEST EXECUTION SUMMARY (Completed in {elapsed:.2f}s)")
    print(f"Total Tests Run: {result.testsRun}")
    print(f"Successes:      {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures:       {len(result.failures)}")
    print(f"Errors:         {len(result.errors)}")
    print("=" * 70)

    if not result.wasSuccessful():
        print("❌ TEST SUITE FAILED!")
        sys.exit(1)
    else:
        print("✅ ALL TESTS PASSED SUCCESSFULLY 100%!")
        sys.exit(0)

if __name__ == "__main__":
    run_all_suites()
