#!/usr/bin/env bash
source /home/bill/.ncaaf/bin/activate
python - <<'PY'
import pandas as pd, numpy as np
p = pd.read_csv('/home/bill/ncaaf/etl/data/pbp/formatted/pbp_edit.csv',
                nrows=400000, low_memory=False)

st = p[p.special_teams_play == 1]
print(f"=== share of plays that are special teams ===")
print(f"  {len(st)}/{len(p)} = {len(st)/len(p):.1%}")

print("\n=== are ST plays included in the efficiency stats? ===")
for col in ('rushing_play','passing_play','offensive_play'):
    if col in st.columns:
        print(f"  ST plays flagged {col}: {int(st[col].sum())}")
print("  -> the 12 modelled stats are all rush/pass, so ST contributes none of them")

print("\n=== but ST plays DO carry EPA ===")
if 'epa' in st.columns:
    e = st['epa'].dropna()
    print(f"  ST plays with EPA: {len(e)}  mean {e.mean():+.4f}  sd {e.std():.3f}")
    print(f"  total |EPA| on ST: {e.abs().sum():,.0f}")
    off = p[p.offensive_play == 1]['epa'].dropna()
    print(f"  total |EPA| on offense: {off.abs().sum():,.0f}")
    print(f"  ST share of all EPA magnitude: {e.abs().sum()/(e.abs().sum()+off.abs().sum()):.1%}")

print("\n=== breakdown by ST play type ===")
if 'play_type_text' in st.columns:
    vc = st['play_type_text'].value_counts().head(10)
    for k, n in vc.items():
        sub = st[st.play_type_text == k]['epa'].dropna()
        print(f"  {str(k)[:28]:<30} n={n:>6}  mean EPA {sub.mean():>+7.3f}  sd {sub.std():>6.3f}")

print("\n=== the variance question: how much does ST swing a game? ===")
if 'epa' in st.columns and 'game_id' in st.columns:
    per_game = st.groupby(['game_id','team_id'])['epa'].sum()
    print(f"  ST EPA per team-game: mean {per_game.mean():+.2f}  sd {per_game.std():.2f}")
    off_pg = p[p.offensive_play == 1].groupby(['game_id','team_id'])['epa'].sum()
    print(f"  offensive EPA per team-game: mean {off_pg.mean():+.2f}  sd {off_pg.std():.2f}")
    print(f"  ST sd as share of offensive sd: {per_game.std()/off_pg.std():.1%}")
PY
