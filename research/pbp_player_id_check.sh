#!/usr/bin/env bash
source /home/bill/.ncaaf/bin/activate
python - <<'PY'
import pandas as pd
p = pd.read_csv('/home/bill/ncaaf/etl/data/pbp/formatted/pbp_edit.csv',
                nrows=4000, low_memory=False)
print(f"=== columns ({len(p.columns)}) ===")
for c in p.columns:
    print(f"  {c}")
PY
