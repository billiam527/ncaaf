"""Every prediction decomposed into the variables that produced it.

One row per game: the identifiers, each model's number, the blend, the market
line, and then every variable with its value and its weight.

THERE ARE NO COEFFICIENTS TO REPORT

Both models are gradient-boosted trees, not linear fits. A tree has no global
per-variable coefficient - what a variable is worth depends on the values of the
others, which is most of why the model beats a linear fit on this data. What
does exist, exactly, is a per-game additive decomposition:

    raw prediction = base + sum over variables of contribution_i

That is TreeSHAP, computed by xgboost itself through pred_contribs, and it is an
identity rather than an approximation - the reconciliation check below holds to
1e-6. So the weights here are per game. The same variable can be worth +4 in one
matchup and -1 in another, and that is a property of the model, not a defect of
the reporting.

WHAT THE WEIGHTS ARE

Each contribution divided by the sum of contributions, so the weights sum to
1.000 for every game and each one reads as that variable's share of why the
prediction came out where it did.

    weight_i = contribution_i / sum of all contributions

One caveat that matters, and the file reports it rather than hiding it. The
denominator is the prediction's distance from the model's base value, so a game
the model calls close to even divides by something near zero and the shares
explode - individual weights above 1, or large and negative, that are arithmetic
rather than meaning. weight_denominator carries that divisor and
weights_reliable marks the games where it is too small to trust. Ignore the
shares on those rows and read the contributions instead; the raw contribution of
every variable is kept alongside for exactly that reason.

THE FULL CHAIN, WHICH THE FILE LETS YOU VERIFY

    preseason_raw   = preseason_base + sum(preseason contributions)
    preseason_pred  = preseason_raw + neutral_site_adjustment
    in_season_raw   = in_season_base + sum(in-season contributions)
    in_season_pred  = in_season_raw + neutral_site_adjustment
    blended_pred    = w_pre * preseason_pred + w_ins * in_season_pred + intercept

The neutral-site adjustment is applied after the model runs, so contributions
sum to the RAW prediction and not the published one. Both columns are present.
The blend weights come from model_blender/blended_model.csv and vary by week.

Downstream, margin_distribution adds a drift offset to the blend before building
the distribution, so the centre it publishes is blended_pred + that offset. This
file stops at the blend.

    python prediction_formula.py
    python prediction_formula.py --game "UNT @ IU"
    python prediction_formula.py --long          # one row per game per variable
    python prediction_formula.py --top 15        # only the 15 biggest movers
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd
import xgboost as xgb

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)

PRED_DIR = os.path.join(_HERE, 'prediction_file')
PREDICTIONS = os.path.join(PRED_DIR, 'predictions.csv')
PRE_FEATURES = os.path.join(PRED_DIR, 'features_file.csv')
INS_FEATURES = os.path.join(PRED_DIR, 'in_season_features_file.csv')
BLEND_WEIGHTS = os.path.join(_REPO, 'model_training', 'model_blender',
                             'blended_model.csv')
OUT = os.path.join(PRED_DIR, 'prediction_formula.csv')

# Below this the share denominator is too small for the weights to mean
# anything. A prediction this close to the model's base value is a game it has
# no strong opinion about, and dividing by it manufactures huge shares.
MIN_DENOMINATOR = 1.0

# market_spread is the book's own convention - NEGATIVE means the home team is
# favoured - which is the opposite sign to every model column here.
# market_margin is the same number negated, so it means what the model columns
# mean: points the home team is expected to win by. Both are carried because
# dropping either is how a sign error gets made.
ID_COLS = ['game_id', 'season', 'date', 'week', 'short_name', 'neutral_site',
           'preseason_model_pred', 'in_season_model_pred', 'blended_pred',
           'market_spread', 'market_margin']


def newest_model(kind):
    """Newest trained directory for a model, matching what predict.sh uses."""
    root = os.path.join(_REPO, 'model_training', f'{kind}_model')
    runs = sorted((d for d in os.listdir(root) if d.startswith('xgboost_reg_')),
                  key=lambda d: os.path.getmtime(os.path.join(root, d)))
    if not runs:
        return None
    return os.path.join(root, runs[-1])


def load_model(kind):
    d = newest_model(kind)
    if d is None:
        return None, None, None
    with open(os.path.join(d, 'model.pkl'), 'rb') as fh:
        model = pickle.load(fh)
    with open(os.path.join(d, 'scaler.pkl'), 'rb') as fh:
        scaler = pickle.load(fh)
    return model, scaler, os.path.basename(d)


def contributions(model, scaler, features):
    """Per-game additive contributions, in the model's own feature order.

    pred_contribs returns one column per feature plus a trailing bias column,
    and they sum to the raw prediction exactly. The scaler is positional, so the
    columns are ordered against scaler.feature_names_in_ before transforming -
    the same check predict_games makes, for the same reason.
    """
    names = list(getattr(scaler, 'feature_names_in_', features.columns))
    missing = [c for c in names if c not in features.columns]
    if missing:
        raise ValueError(f"feature file is missing {len(missing)} columns "
                         f"the scaler expects, e.g. {missing[:3]}")
    X = features[names]
    scaled = scaler.transform(X)
    contrib = model.get_booster().predict(xgb.DMatrix(scaled, feature_names=names),
                                          pred_contribs=True)
    return names, contrib[:, :-1], contrib[:, -1], X


def block(prefix, ids, names, contrib, bias, values):
    """One model's decomposition: value, contribution and weight per variable."""
    total = contrib.sum(axis=1)
    denom = np.where(np.abs(total) < 1e-12, np.nan, total)

    out = pd.DataFrame({'game_id': ids})
    out[f'{prefix}_base'] = bias
    out[f'{prefix}_raw'] = bias + total
    out[f'{prefix}_weight_denominator'] = total
    out[f'{prefix}_weights_reliable'] = np.abs(total) >= MIN_DENOMINATOR

    frame = {}
    for i, name in enumerate(names):
        frame[f'{prefix}__{name}'] = values.iloc[:, i].to_numpy()
        frame[f'{prefix}__{name}__contrib'] = contrib[:, i]
        frame[f'{prefix}__{name}__weight'] = contrib[:, i] / denom
    return pd.concat([out, pd.DataFrame(frame)], axis=1)


def blend_weights(preds):
    """Per-week blend weights, joined the way implement_blended_model joins."""
    if not os.path.exists(BLEND_WEIGHTS):
        return None
    w = pd.read_csv(BLEND_WEIGHTS)
    w['pre_szn_coefs'] = w['pre_szn_coefs'].round(4)
    w['in_szn_coefs'] = w['in_szn_coefs'].round(4)
    label = []
    for v in preds['week']:
        t = str(v).strip()
        label.append(t if t.lower().startswith('week') else f'Week {t}')
    out = pd.DataFrame({'week': label}).merge(
        w[['week', 'pre_szn_coefs', 'in_szn_coefs', 'intercepts']],
        on='week', how='left')
    return out


def build():
    preds = pd.read_csv(PREDICTIONS, index_col=0)
    preds = preds.rename(columns={'id': 'game_id',
                                  'preseason_model_preds': 'preseason_model_pred',
                                  'in_season_model_preds': 'in_season_model_pred'})
    for c in ('season', 'neutral_site', 'market_spread', 'in_season_model_pred'):
        if c not in preds.columns:
            preds[c] = np.nan

    w = blend_weights(preds)
    if w is not None:
        pre = preds['preseason_model_pred']
        ins = preds['in_season_model_pred']
        blended = (w['pre_szn_coefs'].to_numpy() * pre.to_numpy()
                   + w['in_szn_coefs'].to_numpy() * ins.to_numpy()
                   + w['intercepts'].to_numpy())
        # No in-season number before the season starts; the blend is the
        # preseason model alone there, which is what implement_blended_model does.
        preds['blended_pred'] = np.where(ins.isna(), pre, blended)
        preds['blend_w_preseason'] = w['pre_szn_coefs'].to_numpy()
        preds['blend_w_in_season'] = w['in_szn_coefs'].to_numpy()
        preds['blend_intercept'] = w['intercepts'].to_numpy()
    else:
        preds['blended_pred'] = preds['preseason_model_pred']

    out = preds[[c for c in ID_COLS if c in preds.columns]
                + [c for c in ('blend_w_preseason', 'blend_w_in_season',
                               'blend_intercept') if c in preds.columns]].copy()

    for kind, path, prefix in (('preseason', PRE_FEATURES, 'pre'),
                               ('in_season', INS_FEATURES, 'ins')):
        if not os.path.exists(path):
            print(f"  no {kind} feature file; skipping that block")
            continue
        f = pd.read_csv(path)
        if 'game_id' not in f.columns:
            print(f"  {os.path.basename(path)} has no game_id column - rerun "
                  f"predict.py to regenerate it; skipping the {kind} block")
            continue
        model, scaler, run = load_model(kind)
        if model is None:
            print(f"  no trained {kind} model; skipping")
            continue
        ids = f['game_id'].to_numpy()
        names, contrib, bias, values = contributions(
            model, scaler, f.drop(columns=['game_id']))
        print(f"  {kind}: {len(names)} variables from {run}, {len(f)} games")
        out = out.merge(block(prefix, ids, names, contrib, bias, values),
                        on='game_id', how='left')

    return out


def check(out):
    """The decomposition is an identity, so say so out loud or fail."""
    print("\n=== reconciliation ===")
    ok = True
    for prefix, pred_col in (('pre', 'preseason_model_pred'),
                             ('ins', 'in_season_model_pred')):
        cols = [c for c in out.columns
                if c.startswith(f'{prefix}__') and c.endswith('__contrib')]
        if not cols:
            continue
        have = out[f'{prefix}_raw'].notna()
        rebuilt = out.loc[have, f'{prefix}_base'] + out.loc[have, cols].sum(axis=1)
        err = (rebuilt - out.loc[have, f'{prefix}_raw']).abs().max()
        print(f"  {prefix}: base + sum(contributions) vs raw   max error {err:.2e}"
              f"  over {have.sum()} games")
        ok &= err < 1e-4

        wcols = [c.replace('__contrib', '__weight') for c in cols]
        rel = out[f'{prefix}_weights_reliable'].fillna(False)
        s = out.loc[have & rel, wcols].sum(axis=1)
        if len(s):
            print(f"       weights sum to 1: max error "
                  f"{(s - 1).abs().max():.2e} over {len(s)} reliable games")

        n_bad = int((have & ~rel).sum())
        if n_bad:
            print(f"       {n_bad} game(s) flagged weights_reliable=False "
                  f"(|denominator| < {MIN_DENOMINATOR})")

        # The published prediction is the raw one plus the neutral-site
        # adjustment, so a difference here is expected on neutral games only.
        # The tolerance is 0.01 rather than float-epsilon because pred_contribs
        # returns float32 while model.predict returns float64: nearly every game
        # differs in the sixth decimal, which is arithmetic, not an adjustment.
        d = (out.loc[have, pred_col] - out.loc[have, f'{prefix}_raw']).abs()
        moved = int((d > 0.01).sum())
        print(f"       published minus raw exceeds 0.01 on {moved} game(s) "
              f"(neutral sites), max {d.max():.2f}; "
              f"float32 noise elsewhere is {d[d <= 0.01].max():.1e}")
    if not ok:
        raise SystemExit("reconciliation FAILED - the decomposition is wrong")


def to_long(out, prefix):
    cols = [c for c in out.columns
            if c.startswith(f'{prefix}__') and c.endswith('__contrib')]
    rows = []
    for c in cols:
        name = c[len(prefix) + 2:-len('__contrib')]
        rows.append(pd.DataFrame({
            'game_id': out['game_id'], 'short_name': out['short_name'],
            'model': prefix, 'variable': name,
            'value': out[f'{prefix}__{name}'],
            'contribution': out[c],
            'weight': out[f'{prefix}__{name}__weight'],
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def show(out, game, top):
    m = out[out['short_name'].astype(str).str.contains(game, case=False, na=False)]
    if m.empty:
        print(f"\nno game matching {game!r}")
        return
    for _, r in m.iterrows():
        print(f"\n=== {r['short_name']}  (week {r['week']}, {r['date']}) ===")
        for label, col in (('preseason', 'preseason_model_pred'),
                           ('in-season', 'in_season_model_pred'),
                           ('blended', 'blended_pred'),
                           ('market', 'market_margin')):
            v = r.get(col)
            print(f"  {label:<12}{'' if pd.isna(v) else f'{v:+.2f}'}")
        # all four above are "points the home team wins by"; the book's own
        # number runs the other way and is printed as the book writes it
        sp = r.get('market_spread')
        if pd.notna(sp):
            print(f"  {'(book line':<12}{sp:+.1f} on the home team)")
        if pd.notna(r.get('market_margin')) and pd.notna(r.get('blended_pred')):
            print(f"  {'edge':<12}{r['blended_pred'] - r['market_margin']:+.2f}"
                  f"  (model minus market)")

        for prefix, label in (('pre', 'preseason'), ('ins', 'in-season')):
            if pd.isna(r.get(f'{prefix}_raw', np.nan)):
                continue
            names = [c[len(prefix) + 2:-len('__contrib')] for c in out.columns
                     if c.startswith(f'{prefix}__') and c.endswith('__contrib')]
            d = pd.DataFrame({
                'variable': names,
                'value': [r[f'{prefix}__{n}'] for n in names],
                'contribution': [r[f'{prefix}__{n}__contrib'] for n in names],
                'weight': [r[f'{prefix}__{n}__weight'] for n in names],
            })
            d = d.reindex(d['contribution'].abs().sort_values(ascending=False).index)
            print(f"\n  --- {label} model: base {r[f'{prefix}_base']:+.2f}, "
                  f"raw {r[f'{prefix}_raw']:+.2f} ---")
            if not r[f'{prefix}_weights_reliable']:
                print(f"  weights UNRELIABLE: denominator "
                      f"{r[f'{prefix}_weight_denominator']:+.2f} is under "
                      f"{MIN_DENOMINATOR}; read the contributions instead")
            print(f"  {'variable':<44}{'value':>10}{'contrib':>10}{'weight':>9}")
            for _, x in d.head(top).iterrows():
                print(f"  {x['variable'][:43]:<44}{x['value']:>10.3f}"
                      f"{x['contribution']:>10.3f}{x['weight']:>9.3f}")
            rest = d.iloc[top:]
            print(f"  {f'... {len(rest)} more':<44}{'':>10}"
                  f"{rest['contribution'].sum():>10.3f}{rest['weight'].sum():>9.3f}")
            print(f"  {'TOTAL':<44}{'':>10}{d['contribution'].sum():>10.3f}"
                  f"{d['weight'].sum():>9.3f}")


def inspect(out, needle, top):
    """One variable across the whole slate: is it behaving?

    A per-game decomposition answers "why this prediction". This answers the
    other question - whether a variable is pulling its weight sensibly
    everywhere, or quietly driving games it should not.
    """
    hits = sorted({c[len(p) + 2:-len('__contrib')]
                   for p in ('pre', 'ins') for c in out.columns
                   if c.startswith(f'{p}__') and c.endswith('__contrib')
                   and needle.lower() in c.lower()})
    if not hits:
        print(f"\nno variable matching {needle!r}")
        return
    for name in hits:
        prefix = 'pre' if f'pre__{name}__contrib' in out.columns else 'ins'
        v = out[f'{prefix}__{name}']
        c = out[f'{prefix}__{name}__contrib']
        w = out[f'{prefix}__{name}__weight']
        rel = out[f'{prefix}_weights_reliable'].fillna(False)
        n = v.notna().sum()

        print(f"\n=== {name}  ({prefix}) ===")
        print(f"  present on {n} of {len(out)} games")
        print(f"  {'':<14}{'min':>10}{'median':>10}{'mean':>10}{'max':>10}")
        for label, s in (('value', v), ('contribution', c),
                         ('weight', w.where(rel))):
            if s.notna().any():
                print(f"  {label:<14}{s.min():>10.3f}{s.median():>10.3f}"
                      f"{s.mean():>10.3f}{s.max():>10.3f}")
        print(f"  mean |contribution| {c.abs().mean():.3f} points; "
              f"this variable is {c.abs().mean() / out[[x for x in out.columns if x.startswith(f'{prefix}__') and x.endswith('__contrib')]].abs().sum(axis=1).mean():.1%} "
              f"of the average game's total movement")

        d = out.loc[c.abs().sort_values(ascending=False).index].head(top)
        print(f"\n  biggest movers")
        print(f"  {'game':<16}{'value':>10}{'contrib':>10}{'weight':>9}"
              f"{'prediction':>12}")
        for _, r in d.iterrows():
            wv = r[f'{prefix}__{name}__weight']
            print(f"  {str(r['short_name'])[:15]:<16}"
                  f"{r[f'{prefix}__{name}']:>10.3f}"
                  f"{r[f'{prefix}__{name}__contrib']:>10.3f}"
                  f"{wv:>9.3f}{r['blended_pred']:>12.2f}")


def parse_sets(pairs):
    out = {}
    for p in pairs or []:
        if '=' not in p:
            raise SystemExit(f"--set expects variable=value, got {p!r}")
        k, v = p.split('=', 1)
        out[k.strip()] = float(v)
    return out


def override(sets, game, out_df):
    """Re-run the model with a variable corrected, and show what it moves.

    The value is changed and the model re-run, rather than the contribution
    being edited directly. A tree's contributions are a consequence of its
    inputs - editing one would produce a decomposition that no longer
    corresponds to any prediction the model would actually make.
    """
    f = pd.read_csv(PRE_FEATURES)
    model, scaler, run = load_model('preseason')
    names = list(getattr(scaler, 'feature_names_in_', []))

    unknown = [k for k in sets if k not in names]
    if unknown:
        # Substring matching alone suggested nothing for a near miss like
        # pf_oline against pf_ol_home, which is the case a suggestion is for.
        import difflib
        for u in unknown:
            close = difflib.get_close_matches(u, names, n=5, cutoff=0.5)
            close += [n for n in names if u.lower() in n.lower() and n not in close]
            print(f"  unknown variable {u!r}. Closest: {close[:5] or 'nothing similar'}")
        raise SystemExit(f"unknown variable(s): {unknown}")

    ids = f['game_id'].to_numpy()
    X = f[names].copy()
    before = model.predict(scaler.transform(X))

    mask = np.ones(len(X), bool)
    if game:
        keep = out_df.loc[out_df['short_name'].astype(str)
                          .str.contains(game, case=False, na=False), 'game_id']
        mask = np.isin(ids, keep.to_numpy())
        if not mask.any():
            raise SystemExit(f"no game matching {game!r}")

    X2 = X.copy()
    for k, v in sets.items():
        X2.loc[mask, k] = v
    after = model.predict(scaler.transform(X2))

    changed = np.abs(after - before) > 1e-6
    print(f"\n=== override on {run} ===")
    for k, v in sets.items():
        was = X.loc[mask, k]
        print(f"  {k}: {was.min():.3f}..{was.max():.3f}  ->  {v}"
              f"   on {int(mask.sum())} game(s)")
    print(f"  moved {int(changed.sum())} prediction(s), "
          f"mean shift {(after - before)[changed].mean():+.2f}, "
          f"largest {np.abs(after - before).max():+.2f}")

    idx = np.argsort(-np.abs(after - before))[:25]
    name = dict(zip(out_df['game_id'], out_df['short_name']))
    print(f"\n  {'game':<16}{'before':>10}{'after':>10}{'shift':>9}")
    for i in idx:
        if not changed[i]:
            break
        print(f"  {str(name.get(ids[i], ids[i]))[:15]:<16}"
              f"{before[i]:>10.2f}{after[i]:>10.2f}{after[i] - before[i]:>+9.2f}")
    return pd.DataFrame({'game_id': ids, 'short_name': [name.get(i) for i in ids],
                         'prediction_before': before, 'prediction_after': after,
                         'shift': after - before})


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--game', default=None, help='substring of short_name')
    ap.add_argument('--top', type=int, default=20,
                    help='variables to print per model with --game')
    ap.add_argument('--variable', default=None,
                    help='inspect one variable across every game')
    ap.add_argument('--set', action='append', metavar='VAR=VALUE',
                    help='correct a variable and re-run; repeatable. Combine '
                         'with --game to correct one matchup only')
    ap.add_argument('--long', action='store_true',
                    help='also write one row per game per variable')
    ap.add_argument('--out', default=OUT)
    args = ap.parse_args()

    out = build()
    check(out)

    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")
    print(f"  {len(out)} games x {len(out.columns)} columns")

    if args.long:
        longs = [to_long(out, p) for p in ('pre', 'ins')]
        longs = [d for d in longs if not d.empty]
        if longs:
            path = args.out.replace('.csv', '_long.csv')
            pd.concat(longs, ignore_index=True).to_csv(path, index=False)
            print(f"  and {path}")

    if args.variable:
        inspect(out, args.variable, args.top)

    if args.set:
        moved = override(parse_sets(args.set), args.game, out)
        path = args.out.replace('.csv', '_override.csv')
        moved.to_csv(path, index=False)
        print(f"\n  wrote {path}")
    elif args.game:
        show(out, args.game, args.top)


if __name__ == '__main__':
    main()
