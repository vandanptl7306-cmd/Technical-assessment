import pandas as pd
from prompt_evaluator.data_loader import load_dataset
from prompt_evaluator.evaluator import AnswerEvaluator
from prompt_evaluator.report_generator import compute_metrics

# 1. Test dataset loading
df, meta = load_dataset("questions.csv")
assert len(df) == 300, f"Expected 300 rows, got {len(df)}"
assert meta["is_valid"] is True
print("Test 1: Dataset loading and validation passed.")

# 2. Test evaluator
evaluator = AnswerEvaluator()
results_df = evaluator.evaluate_dataframe(df, tie_threshold=2.5)
assert len(results_df) == 300
assert "new_risk_severity" in results_df.columns
print("Test 2: Evaluator execution passed.")

# 3. Test verdicts.csv
df_v = pd.read_csv("outputs/verdicts.csv")
assert df_v.shape == (300, 2), f"Invalid verdicts shape: {df_v.shape}"
assert list(df_v.columns) == ["id", "better"], f"Invalid columns: {df_v.columns}"
assert set(df_v["better"].unique()) == {"current", "new", "tie"}
assert len(df_v) == 300
print("Test 3: verdicts.csv format and counts verified (exactly 300 rows).")

# 4. Test Safety-First Decision
metrics = compute_metrics(results_df, tie_threshold=2.5)
assert metrics["recommendation"] == "DO NOT SHIP"
assert metrics["critical_security_count"] > 0
print(f"Test 4: Safety-First Decision verified: {metrics['recommendation']} ({metrics['critical_security_count']} critical security risks detected).")

# 5. Verify scores.csv
df_s = pd.read_csv("outputs/scores.csv")
assert len(df_s) == 300
crit_count = (df_s["new_risk_severity"] == "Critical").sum()
high_count = (df_s["new_risk_severity"] == "High").sum()
med_count = (df_s["new_risk_severity"] == "Medium").sum()
print(f"Test 5: Risk classification in scores.csv: {crit_count} Critical, {high_count} High, {med_count} Medium.")

print("\nALL VERIFICATION TESTS PASSED 100%!")
