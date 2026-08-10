# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 15:36:32 2024

@author: wfish

Learns per-week weights for combining the preseason and in-season model
predictions, from the historical per-week files batch_prediction/temp/ holds
(written by predict.py --model_blender True).

Blending is worth roughly 1.5 MAE over either model alone; tuning the weights
is worth about 0.16 on top of that. A flat 50/50 average therefore captures
most of the available gain, so validate() checks the fitted weights actually
beat it out of sample before you trust them.
"""
import os
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

XS = ['preseason_model_preds', 'in_season_model_preds']
Y = 'home_score_differential'

# Below this many games a per-week fit is noise; fall back to the pooled fit.
# Week 16 previously learned an in-season coefficient of -0.015 from 4 games.
MIN_GAMES_FOR_WEEKLY_FIT = 40


def pull_in_data(prediction_file_loc):
    """Read temp/<week>_<season>.csv into one frame with week and season."""
    temp_dir = os.path.join(prediction_file_loc, 'temp')
    frames = []
    for fname in os.listdir(temp_dir):
        if not fname.endswith('.csv'):
            continue
        m = re.match(r'^(.*)_(\d{4})\.csv$', fname)
        if not m:
            continue
        week, season = m.group(1), int(m.group(2))
        df = pd.read_csv(os.path.join(temp_dir, fname), index_col=0)
        df['week'] = week
        df['season'] = season
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"no per-week files in {temp_dir}. Generate them with:\n"
            f"  cd batch_prediction && python predict.py ... --model_blender True")

    data = pd.concat(frames, ignore_index=True)
    return data.dropna(subset=XS + [Y])


def _fit(df):
    """Non-negative least squares blend. Returns (pre, in, intercept).

    positive=True matters: a negative weight on either model is never
    meaningful here, and unconstrained OLS produced them on sparse weeks.
    """
    model = LinearRegression(positive=True).fit(df[XS], df[Y])
    return float(model.coef_[0]), float(model.coef_[1]), float(model.intercept_)


def _week_sort_key(week):
    m = re.search(r'(\d+)', str(week))
    return (0, int(m.group(1))) if m else (1, 0)


def compare_results(data):
    """Fit weights per week, pooling sparse weeks and forcing Week 1."""
    pooled = _fit(data)

    rows = []
    for week in sorted(data['week'].unique(), key=_week_sort_key):
        df = data[data['week'] == week]

        if week == 'Week 1':
            # No games have been played, so there is no in-season signal.
            pre, insz, intercept = 1.0, 0.0, 0.0
        elif len(df) < MIN_GAMES_FOR_WEEKLY_FIT:
            pre, insz, intercept = pooled
        else:
            pre, insz, intercept = _fit(df)

        rows.append({'week': week, 'pre_szn_coefs': pre,
                     'in_szn_coefs': insz, 'intercepts': intercept,
                     'n_games': len(df)})

    blended_model = pd.DataFrame(rows)

    # Week 1 must exist even with no history for it: implement_blended_model.py
    # joins predictions to this table, and a missing week silently drops those
    # games from the output.
    if 'Week 1' not in set(blended_model['week']):
        blended_model = pd.concat([
            pd.DataFrame([{'week': 'Week 1', 'pre_szn_coefs': 1.0,
                           'in_szn_coefs': 0.0, 'intercepts': 0.0, 'n_games': 0}]),
            blended_model], ignore_index=True)

    return blended_model


def validate(data):
    """Walk-forward check: are the fitted weights better than a 50/50 average?

    Each season is scored by weights fit only on earlier seasons.
    """
    seasons = sorted(data['season'].unique())
    rows = []
    for season in seasons[3:]:               # need some history to fit on
        train = data[data['season'] < season]
        test = data[data['season'] == season]
        weights = compare_results(train).set_index('week')

        pre = test['preseason_model_preds'].values
        insz = test['in_season_model_preds'].values
        actual = test[Y].values

        blended = np.empty(len(test))
        for i, week in enumerate(test['week'].values):
            if week in weights.index:
                w = weights.loc[week]
            else:
                w = weights.iloc[0]
            blended[i] = (w['pre_szn_coefs'] * pre[i]
                          + w['in_szn_coefs'] * insz[i] + w['intercepts'])

        rows.append({
            'season': season,
            'n': len(test),
            'preseason': np.abs(pre - actual).mean(),
            'in_season': np.abs(insz - actual).mean(),
            'fixed_50': np.abs(0.5 * pre + 0.5 * insz - actual).mean(),
            'blended': np.abs(blended - actual).mean(),
        })

    res = pd.DataFrame(rows)
    if res.empty:
        return res

    weighted = {c: (res[c] * res['n']).sum() / res['n'].sum()
                for c in ('preseason', 'in_season', 'fixed_50', 'blended')}

    print("\nout-of-sample MAE (weights fit on earlier seasons only):")
    print(res.round(2).to_string(index=False))
    print("\npooled:")
    for k, v in weighted.items():
        print(f"  {k:<10} {v:6.3f}")

    edge = weighted['fixed_50'] - weighted['blended']
    if edge > 0:
        print(f"\n  fitted weights beat a 50/50 average by {edge:.3f} MAE")
    else:
        print(f"\n  WARNING: fitted weights are {-edge:.3f} MAE WORSE than a 50/50 "
              f"average. Prefer the flat blend until this is understood.")
    return res


if __name__ == '__main__':
    data = pull_in_data('../../batch_prediction/')
    print(f"{len(data)} games, seasons "
          f"{int(data.season.min())}-{int(data.season.max())}")

    validate(data)

    blended_model = compare_results(data)
    blended_model.to_csv('blended_model.csv')
    print(f"\nwrote blended_model.csv ({len(blended_model)} weeks)")
