#!/usr/bin/env bash
# Runs the whole analysis after data collection: notebooks 02 -> 05, executed in place
# (outputs are saved into the notebooks), tables and figures written to results/.
# Usage:  ./run_analysis.sh            (refuses to run while data collection is unfinished)
#         ./run_analysis.sh --force    (run anyway on the data collected so far)
set -euo pipefail
cd "$(dirname "$0")"

PY=.venv/bin/python
JUPYTER=.venv/bin/jupyter

if [[ "${1:-}" != "--force" ]]; then
  "$PY" - <<'EOF'
import json, os, sys
from pathlib import Path
pid = Path("data/logs/collector.pid")
if pid.exists():
    try:
        os.kill(int(pid.read_text()), 0)
        sys.exit("Data collection is still running. Wait for it to finish (or use --force).")
    except ProcessLookupError:
        pass
sample = Path("data/interim/sample.csv")
if not sample.exists():
    sys.exit("No sample yet (data/interim/sample.csv). Data collection has not reached the details stage.")
import pandas as pd
repos = pd.read_csv(sample, usecols=["full_name"]).full_name.nunique()
done = len(list(Path("data/raw/details").glob("*.done")))
if done < repos:
    sys.exit(f"Details collected for {done}/{repos} repositories only. Finish collection first (or use --force).")
print(f"Data complete: details for {done}/{repos} repositories.")
EOF
fi

echo "=== quality checks (results/qa/)"
"$PY" src/qa_checks.py

for nb in 02_data_preparation 03_rq1_analysis 04_rq2_analysis 05_rq3_analysis; do
  echo "=== running notebooks/$nb.ipynb"
  (cd notebooks && "../$JUPYTER" nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=-1 "$nb.ipynb")
done

echo
echo "Done. Tables: results/tables/   Figures: results/figures/"
ls results/tables results/figures
