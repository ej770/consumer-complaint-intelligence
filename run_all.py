"""Run the full pipeline end to end: python run_all.py  (about 20 minutes on a laptop)."""
import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
STEPS = ["01_clean.py", "02_sql_analysis.py", "03_text_ai.py", "04_models.py",
         "05_export.py", "06_figures.py", "07_build_dashboard.py"]

for step in STEPS:
    start = time.time()
    print(f"\n>>> {step}", flush=True)
    subprocess.run([sys.executable, step], cwd=SRC, check=True)
    print(f"<<< {step} finished in {time.time() - start:.0f}s", flush=True)
