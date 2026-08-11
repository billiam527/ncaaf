#!/usr/bin/env bash
source /home/bill/.ncaaf/bin/activate
cd /home/bill/ncaaf/batch_prediction

echo "=== how many temp files, and when written? ==="
ls -la temp/ | head -5
echo "  total: $(ls temp/*.csv 2>/dev/null | wc -l)"
echo "  distinct write dates:"
ls -l --time-style=+%m-%d_%H:%M temp/*.csv | awk '{print $6}' | sort | uniq -c | sed 's/^/    /'

echo
python - <<'PY'
import pandas as pd, glob, re
rows = []
for f in glob.glob('temp/*.csv'):
    m = re.match(r'^temp/(.*)_(\d{4})\.csv$', f)
    if not m:
        continue
    d = pd.read_csv(f, index_col=0)
    rows.append({'season': int(m.group(2)), 'week': m.group(1), 'n': len(d),
                 'cols': tuple(d.columns)})
df = pd.DataFrame(rows)
print("column sets present across temp files:")
for cols, g in df.groupby('cols'):
    print(f"  seasons {sorted(g.season.unique())}  n_files={len(g)}")
    print(f"    {list(cols)}")
PY

echo
echo "=== spot-check one contaminated season vs one clean one ==="
python - <<'PY'
import pandas as pd, glob
for season in (2021, 2025):
    fs = glob.glob(f'temp/*_{season}.csv')
    d = pd.concat([pd.read_csv(f, index_col=0) for f in fs], ignore_index=True)
    d = d.dropna(subset=['preseason_model_preds', 'home_score_differential'])
    err = (d.preseason_model_preds - d.home_score_differential).abs()
    print(f"\n{season}: n={len(d)} preseason MAE={err.mean():.2f}")
    print(d[['short_name', 'preseason_model_preds', 'home_score_differential']].head(5).to_string(index=False))
    print(f"  corr(pred, actual) = {d.preseason_model_preds.corr(d.home_score_differential):.4f}")
PY
