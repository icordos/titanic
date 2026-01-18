#!/usr/bin/env bash
set -euo pipefail

INPUT="models/day4_ensembles/ensemble_lowcorr_historical_stacker.csv"
OUTPUT="models/day4_ensembles/ensemble_lowcorr_historical_stacker_trimmed.csv"

python - "$INPUT" "$OUTPUT" <<'PY'
import sys
import pandas as pd

inp, out = sys.argv[1], sys.argv[2]
df = pd.read_csv(inp)[["PassengerId", "Transported"]]
df.to_csv(out, index=False)
print(f"Trimmed submission saved to {out}")
PY

#kaggle competitions submit -c spaceship-titanic -f "$OUTPUT" -m "Day1 stacker"
