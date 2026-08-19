# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 15:36:32 2024

@author: wfish

Learns per-week weights for combining the preseason and in-season model
predictions, from the historical per-week files batch_prediction/temp/ holds.

THOSE FILES MUST COME FROM WALK-FORWARD, NOT FROM predict.py

Generate them with walk_forward.generate_expanding_predictions, which retrains
both models for every season on the seasons before it. Scoring a historical
season with today's models is recall, not prediction: on 2025 the current
preseason model returns MAE 10.38 that way, against 13.90 when it has genuinely
not seen the season. Weights fitted on recall load up on whichever model did the
memorising.

The weights that stood here until 2026-08-18 were fitted on a history carrying
two faults at once - an in-season model reading its own season's final stats,
and seasons scored by models trained on them. Refitting on an honest history
moved every week, and moved TOWARDS the in-season model, by +0.20 to +0.48:

    weeks 1-6   mean in-season weight  0.35 -> 0.58
    weeks 7+                           0.60 -> 0.87

That direction was not the expected one. The reading that fits the evidence is
that the old history let the PRESEASON model recite, so it drew weight it had
not earned.

WHAT BLENDING IS WORTH

Out of sample, each season scored by weights fitted only on earlier ones, over
4,419 games from 2020-2025:

    preseason alone   13.780
    in-season alone   13.789
    flat 50/50        13.114
    fitted weights    12.769

Blending is worth about 1.0 MAE over either model alone, and tuning the weights
a further 0.345 - larger than the 0.16 recorded here before, because the two
models are now genuinely complementary rather than two views of the same leaked
information. Their predictions correlate +0.484, against +0.788 when the
in-season model was also given preseason features.

validate() re-runs that comparison. It is there because a flat average is a
serious competitor and the fitted weights have to earn their place.
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
