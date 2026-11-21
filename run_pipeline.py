import os
import shutil
from prompt_evaluator.report_generator import run_full_pipeline

print("Running end-to-end pipeline...")
results_df, metrics, paths = run_full_pipeline(
    dataset_path="questions.csv",
    output_dir="prompt_evaluator/outputs",
    tie_threshold=2.5
)

# Also copy outputs to root outputs/ directory
os.makedirs("outputs", exist_ok=True)
for fname in ["verdicts.csv", "scores.csv", "evaluation_report.md"]:
    src = os.path.join("prompt_evaluator/outputs", fname)
    dst = os.path.join("outputs", fname)
    shutil.copy2(src, dst)

print("\nPipeline execution complete!")
print("Artifacts generated:")
for k, v in paths.items():
    print(f"  {k}: {v} (size: {os.path.getsize(v)} bytes)")

# Verify verdicts.csv
verdicts = open("prompt_evaluator/outputs/verdicts.csv").readlines()
print(f"\nverdicts.csv line count (including header): {len(verdicts)}")
print(f"Header: {verdicts[0].strip()}")
print(f"Sample row 1: {verdicts[1].strip()}")
print(f"Sample row 300: {verdicts[300].strip()}")

# Verify exactly two columns
import pandas as pd
v_df = pd.read_csv("prompt_evaluator/outputs/verdicts.csv")
print(f"verdicts.csv shape: {v_df.shape}")
print(f"verdicts.csv columns: {v_df.columns.tolist()}")
print(f"verdicts.csv value counts:\n{v_df['better'].value_counts()}")
