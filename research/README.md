# research/

One-off analysis scripts that back decisions made about the model. These are not
part of the pipeline — nothing in `etl/`, `model_training/`, or `batch_prediction/`
imports them. They are kept so the measurements behind a decision can be re-run
rather than re-argued.

Each script activates the venv itself and can be run directly:

```bash
./research/residual_signal.sh
```

Generated CSV outputs land in `analysis/`, which is gitignored.

## Home field advantage

| Script | What it measures | Finding |
|---|---|---|
| `residual_signal.sh` | Whether league-average HFA is already absorbed by the model, plus rest and early-season residual structure | Average HFA is captured. Rest and early-season are not — see below |
| `hfa_power_analysis.sh` | Smallest per-venue HFA effect the data could detect | Venue-level HFA is **not** distinguishable from noise. τ ≈ 0 by two estimators; split-half correlation −0.177 |
| `hfa_by_conference.sh` | Same estimator pooled to conference level, where n is large enough to have power | True spread across conferences is only 0.45 points. Not worth modelling |
| `neutral_site_effect.sh` | Margin behaviour at neutral sites | Mean is shifted, and is corrected by `apply_neutral_site_adjustment` |
| `neutral_site_variance.sh` | Whether neutral games also need their own sigma | **No.** The raw margin spread is narrower (18.80 vs 20.67) but that is the model correctly predicting closer games. The residual spread ratio is 0.909 with a 95% CI of [0.793, 1.022] on n=140 — not distinguishable from 1 |
| `residual_variance_by_segment.sh` | Residual bias and spread by part of season | Early season is genuinely wider; champ week and postseason are not. See below |

Conclusion: per-venue HFA is closed, and so is a neutral-site sigma. Do not
revisit either without new data.

## Variance by part of season

Measured on 6,245 walk-forward games, residual spread against the mid-season
baseline:

| segment | n | bias | resid sd | ratio vs mid | 95% CI | verdict |
|---|---|---|---|---|---|---|
| early (wks 2-5) | 1,559 | +1.52 | 19.49 | 1.135 | [1.088, 1.184] | **real** |
| mid (wks 6-13) | 4,272 | −0.29 | 17.17 | — | — | baseline |
| champ week (14) | 319 | −2.71 | 16.53 | 0.962 | [0.885, 1.033] | not distinguishable |
| postseason (15-16) | 95 | −2.59 | 17.26 | 1.000 | [0.840, 1.151] | not distinguishable |

Those numbers are measured on `in_season_model_preds`. **They do not survive into
the published prediction**, which is the blend:

| wk | n | preseason bias | in-season bias | blend bias |
|---|---|---|---|---|
| 2 | 139 | +4.80 | +7.56 | **−0.23** |
| 3 | 415 | +0.17 | +2.28 | **−0.07** |
| 4 | 495 | +0.45 | −0.06 | **−0.10** |
| 5 | 510 | +0.42 | +0.38 | **−0.00** |

Early-vs-mid on the blend: bias −0.07 against −0.02, sd ratio 1.029 with a CI of
[0.984, 1.074]. Per-week blend weights are the mechanism that handles this, and
they handle it — week 2 gets `pre_szn 1.461 / in_szn 0.572`, which amplifies
exactly enough to offset the components' under-prediction. No early-season
correction is warranted.

The general lesson: measure on the column that actually gets published. Both the
neutral-site sigma and the early-season regime looked real on a component model
and vanished on the blend.

Note the confounding between the neutral-site and postseason *mean* effects:
most week 14-16 games are at neutral sites, so the −2.7 champ-week bias and the
−2.6 postseason bias are largely the same games as the neutral-site correction
already being applied. Do not apply both.

Postseason games are present in the walk-forward — `week_num` runs 2 to 16 and
the `week_num < 90` filter in `margin_distribution.load_history` drops nothing.
There is no postseason blind spot.

## FBS vs FCS

| Script | What it measures | Finding |
|---|---|---|
| `fcs_gap.sh` | Share of the slate involving an FCS team, and how those teams are rated | 42 of 471 games (9%). 98 FCS teams rated off a median of **1 game** |
| `fcs_vs_history.sh` | Model prediction vs historical outcome for FBS-v-FCS | Model says +10.8, history says +25.0 — a **14.2 point** error. Zero such games appear in any backtest |

This is the largest known error in the model.

## Margin distribution and calibration

| Script | What it measures | Finding |
|---|---|---|
| `key_numbers.sh` | Frequency of each absolute margin | 3 occurs 2.29x baseline, 7 at 2.13x, 9 at 0.39x. 34.5% of games land on 3/7/10/14/17/21 |
| `key_numbers_by_era.sh` | Whether the key-number curve is stable over time | Stable enough to fit on pooled history |
| `margin_distribution_shape.sh` | Residual quantiles vs normal | Justifies the nfelo-style approach in `margin_distribution.py` |
| `calibration_walkthrough.sh` | Raw prediction vs realised outcome, by band | Isotonic calibration improves MAE 14.42 → 14.00 and Brier 0.2061 → 0.2031 — **in sample**. Superseded by `calibrator_value.sh`, which reverses the conclusion out of sample |
| `calibration_centring.sh` | Centring the distribution on the prediction vs on zero | Centring on the prediction is required — the uncentred version returned a degenerate mode of +9 for a +35 prediction |
| `ats_edge.sh` | Win rate against the closing spread | **No edge.** 50.6 / 50.5 / 49.9% across raw / linear / isotonic over 4,194 bets, against a 52.4% break-even. Closed |
| `calibration_fit_column.sh` | Whether the calibrator is fitted on the column inference centres on | It was not. Fitted on `in_season_model_preds`, applied to `blended_prediction`. Worst probability band was off by **8.9%**; matching the columns brings it to 2.0%. Fixed in `margin_distribution.py` |
| `calibration_walk_forward.sh` | The whole distribution chain refitted per season on earlier seasons only | `--validate` is in-sample and flatters itself: worst band 0.8% against **4.0%** honestly. Sigma is fine — 15.85 fitted against 16.07 realised. The fault was a **+0.95 point centre bias** from rising home advantage (51.0% of games in 2020 to 59.3% in 2025), which `DRIFT_LOOKBACK` now corrects to +0.41 |
| `calibrator_value.sh` | Whether the isotonic calibrator helps out of sample | **It does not.** Damping it toward the identity improves MAE and Brier monotonically, and the limit of that sweep is switching it off: MAE 12.730 → 12.659, Brier 0.1846 → 0.1839, winning on MAE in all three held-out seasons. Its adopted gain was measured in sample on the leaked history. Now off by default; `--calibrate` restores it |

## Model internals

| Script | What it measures | Finding |
|---|---|---|
| `model_coefficients.sh` | Feature names and weights for both models | Feature order differs between models — preseason is 36 away then 36 home, in-season is interleaved |
| `intercept_bias.sh` | Systematic bias in the intercept | — |
| `baseline_comparison.sh` | Model vs constant-prediction floor | Floor is ~15.9 MAE; blended model reaches 13.28 |
| `blender_leak_diagnosis.sh` | Why blender weights looked impossibly good | Weights had been fit on in-sample predictions — preseason correlated 0.98 with actuals. Regenerating from expanding-window predictions dropped that to 0.29–0.51 |
| `backtest_reconciliation.sh` | Whether backtest inputs match training inputs | Caught a missing FBS filter worth ~0.3 MAE |
| `preseason_lag_ablation.py` | Whether `_FY-1` and `_FY-2` prior seasons earn their place | Run via the ablation output in `analysis/preseason_lag_ablation.csv` |
| `encoding_experiment.py` | How the preseason features should be constructed: home/away separately, differences, decayed lags, cross-unit matchups | **No encoding produces a betting edge** — best ATS anywhere is 50.6% against a 52.4% break-even on 2,300 bets. Differentials beat the current 104-column encoding on MAE (13.794 vs 13.834) and ATS (50.1% vs 49.4%) and win 7 seasons to 0, but inside one standard error — the case is 52 columns doing the work of 104, plus fixing the diagonal-boundary artefact that made pair contributions read backwards. Decayed lags cost nothing and cut 52 columns to 28, so recency can be enforced by construction for free. The run-game matchup won a linear test (R² 0.136 vs 0.095) and gained nothing in the real model |
| `tier_hfa.sh` | Whether home advantage differs by conference tier, with Notre Dame and the independents separated and the Pac-12 held apart | **It does not.** Within conference, where home and away schedules balance, P4 is 5.14 and G5 is 5.33 — a gap of 0.19 against se 0.64. The raw tier figures both read 8.73 and are inflated by buy games: a P4 side hosts G5 opponents and rarely visits them, so home-minus-away carries opponent quality with the venue |
| `venue_features.py` | Travel, elevation, capacity and dome against the walk-forward residual | Travel +1.38 points per 1,000 miles (t 2.66), surviving a control for the P4-hosting-G5 mismatch. But the slope is carried by the **0–200 mile band at −1.87** — home teams underperform against *nearby* opponents, which reads as rivalry, not fatigue. Elevation one-sided: climbing +2.12, descending +1.11, both positive, so not the altitude story either |
| `venue_feature_experiment.py` | Whether those features survive as real model features | **They do not.** Travel is worth 0.022 MAE walk-forward and ATS moves 0.5pp on 4,500 bets where se is 0.75pp. Better than base in 6 of 7 seasons, so possibly real and definitely too small to ship. The in-sample t=2.66 was multiplicity across ~8 candidates — the same pattern as the isotonic calibrator |
| `in_season_hyperparams.sh` | Whether the in-season hyperparameters should be re-derived for 72 features | **The tuner was the problem, not the parameters.** `tune_hyperparams.inseason_season` still joined `season_summaries` on `(team_id, season)` — the leak the model no longer has — so `--search` tuned the old 12-feature leaking problem. Fixed. The parameters themselves survive re-searching: 61 stock configs beat the incumbent by 0.043 (t=1.97 as the argmax of 61, so nothing), and a 24-config sweep of the slow region beat it 15 times in 24, mean −0.011. Library defaults lose by **1.265 at t=9.87** — tuning matters hugely, re-tuning does not |

## Play-by-play data audits

| Script | What it measures | Finding |
|---|---|---|
| `epa_sign_audit.sh` | Whether EPA is signed correctly by play category | Offensive EPA is correct (interceptions −4.11, per-game EPA vs margin r = +0.597). **Special teams EPA is wrong** — made FGs score −4.50 and punts +2.53, because possession change and scoring are not handled |
| `special_teams_share.sh` | How much of the game special teams represents | 13.5% of plays but 22.3% of total \|EPA\| magnitude. Per-game ST EPA sd is 9.2 vs offence 13.9. None of it reaches the model — ST plays carry no rush/pass flag |
| `pbp_play_types.sh` | Sample of `play_type_text` values by category | Reference for parsing |
| `pbp_player_id_check.sh` | Whether player identifiers exist anywhere in the pbp file | **They do not.** Any player-level model is blocked on collection, not modelling |

## A note on paths

The pbp file has two copies and they are not the same:

- `etl/data/pbp/formatted/pbp_edit.csv` — a stale Aug 2025 snapshot with EPA largely unpopulated
- `etl/format/format_espn_pbp/temp/pbp_edit.csv` — what `etl/summarize/temp/pbp.csv` symlinks to, and what the model actually uses

Scripts here read the second. Reading the first will silently produce all-zero EPA.
