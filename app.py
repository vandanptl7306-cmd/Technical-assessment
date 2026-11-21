"""
Root entry point for Prompt Evaluator Streamlit app.
Allows running `streamlit run app.py` from workspace root directly.
"""
import os
import sys

# Add directories to path
root_dir = os.path.dirname(os.path.abspath(__file__))
sub_dir = os.path.join(root_dir, "prompt_evaluator")
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if sub_dir not in sys.path:
    sys.path.insert(0, sub_dir)

# Execute the main app script
app_path = os.path.join(sub_dir, "app.py")
with open(app_path, "r", encoding="utf-8") as f:
    code = compile(f.read(), app_path, "exec")
    exec(code, globals())
