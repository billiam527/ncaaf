#!/usr/bin/env python3
"""Corner or safety for every defensive back, including the unlabelled half.

WHY THIS EXISTS

CFBD carries the position a programme reported, and programmes disagree about
how much detail to give: 54% of defensive backs are filed as the coarse 'DB'
while the rest are split into 'S' and 'CB'. The choice is a reporting habit,
not a fact about the player - 70% of team-seasons are mixed, so it is not even
consistent within a roster. That leaves most of the population without a
distinction that plainly exists.

The labelled half trains a classifier for the unlabelled half. They are close
but not identical: mean tackles 17.15 against 15.81, passes defensed 1.66
against 1.50, so programmes that report the fine label carry slightly more
productive players and the training set is mildly the better half.

WHAT SEPARATES THEM

Passes defensed and weight, level at the margin. Dropping either costs the
model about 0.07 of AUC, against 0.018 for tackles - but weight scores 0.758
alone and passes defensed only 0.584, so most of what weight knows is already
carried by tackles while passes defensed is not duplicated anywhere.
Interceptions are worth nothing at all - alone they score AUC 0.511, a coin
flip, and 63.7% of corners against 65.8% of safeties record none in a season.
Safeties convert more of their touches into picks (0.283 per pass defensed
against 0.238) and have far fewer touches; it very nearly cancels.

HOW GOOD IT IS

AUC 0.838 and 75.8% correct on held-out players with real measurements,
against a 55% base rate. That is good enough to label a row on a page and not
good enough to feed a rating: tested as a grading split it moved team-level
prediction by -0.0028 (t -1.10), so nothing here is used for anything but
display.

It used to read 0.868 and 79.4%. Building the defender file from the box score
rather than from parsed play text added 5,565 defensive backs the parser never
named, and they are overwhelmingly backups - 11.6 career tackles against 66.2,
0.6 passes defensed against 6.8. Scored on the men it used to cover the same
classifier reads 0.880, better than the figure it replaces. The number fell
because the population stopped excluding half the position, not because the
model got worse.

Usage:
    python db_position.py --out results/db_position.csv
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

_HERE = os.path.dirname(os.path.abspath(__file__))
PRODUCTION = os.path.join(_HERE, 'results', 'defensive_production.csv')
ROSTER = os.path.join(_HERE, '..', 'collect', 'collect_cfbd_players', 'temp',
                      'cfbd_roster.csv')
SAFETY = {'S', 'FS', 'SS'}
CORNER = {'CB'}
BODY = ['height', 'weight']
PROD = ['tot_box', 'tkl_pg', 'pd_best', 'pd_pg', 'tfl_box']


def build():
    d = pd.read_csv(PRODUCTION, low_memory=False)
    d = d[d['group'] == 'DB'].copy()
    d['pid'] = d['pid'].astype(str)
    for c in ('tot_box', 'pd_best', 'intercept', 'tfl_box'):
        d[c] = pd.to_numeric(d[c], errors='coerce').fillna(0.0)

    agg = d.groupby('pid').agg(
        tot_box=('tot_box', 'sum'), pd_best=('pd_best', 'sum'),
        tfl_box=('tfl_box', 'sum'), seasons=('season', 'nunique'),
        pos=('pos', 'first'), who=('who', 'last'))
    agg['tkl_pg'] = agg['tot_box'] / agg['seasons']
    agg['pd_pg'] = agg['pd_best'] / agg['seasons']

    ros = pd.read_csv(ROSTER, low_memory=False)
    ros['pid'] = ros['id'].astype(str)
    body = (ros.dropna(subset=BODY).sort_values('season').groupby('pid')
               .agg(height=('height', 'last'), weight=('weight', 'last')))
    agg = agg.join(body)

    agg['role'] = np.where(agg['pos'].isin(SAFETY), 'S',
                           np.where(agg['pos'].isin(CORNER), 'CB', None))
    agg['source'] = np.where(agg['role'].notna(), 'cfbd', None)
    known = agg['role'].notna()
    print(f"  {len(agg):,} defensive backs; {known.sum():,} labelled by CFBD, "
          f"{(~known).sum():,} filed only as DB")

    # Two models, because height and weight are missing for some players and a
    # median stand-in measured worse than simply not using the body at all.
    agg['p_safety'] = np.nan
    for feats, tag in ((PROD + BODY, 'production and body'), (PROD, 'production')):
        ok = agg[feats].notna().all(axis=1)
        tr = agg[known & ok]
        if len(tr) < 200:
            continue
        y = (tr['role'] == 'S').astype(int).to_numpy()
        pipe = make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=3000))
        cvp = cross_val_predict(pipe, tr[feats].to_numpy(float), y, cv=5,
                                method='predict_proba')[:, 1]
        auc = roc_auc_score(y, cvp)
        acc = ((cvp > 0.5).astype(int) == y).mean()
        pipe.fit(tr[feats].to_numpy(float), y)
        need = (~known) & ok & agg['p_safety'].isna()
        agg.loc[need, 'p_safety'] = pipe.predict_proba(
            agg.loc[need, feats].to_numpy(float))[:, 1]
        agg.loc[need, 'role'] = np.where(
            agg.loc[need, 'p_safety'] > 0.5, 'S', 'CB')
        agg.loc[need, 'source'] = 'model'
        print(f"  {tag:<22} AUC {auc:.3f}  accuracy {acc:.1%}  "
              f"-> labelled {int(need.sum()):,}")

    left = agg['role'].isna()
    if left.any():
        agg.loc[left, 'role'] = 'CB'
        agg.loc[left, 'source'] = 'default'
        print(f"  {int(left.sum()):,} with too little to judge, defaulted to CB")

    out = agg[['who', 'role', 'source', 'p_safety']].reset_index()
    print(f"\n  {out['role'].value_counts().to_dict()}")
    print(f"  by source: {out['source'].value_counts().to_dict()}")
    conf = out['p_safety'].notna() & (
        (out['p_safety'] < 0.2) | (out['p_safety'] > 0.8))
    n_model = (out['source'] == 'model').sum()
    print(f"  of the modelled ones, {conf.sum() / max(n_model, 1):.1%} are "
          f"confident (p below 0.2 or above 0.8)")
    return out


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(_HERE, 'results',
                                                  'db_position.csv'))
    a = ap.parse_args()
    o = build()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    o.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}  ({len(o):,} players)")
