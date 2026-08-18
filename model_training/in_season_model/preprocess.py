# -*- coding: utf-8 -*-
"""Build the in-season training frame from what had happened BEFORE each game.

Created on Sat Jul 13 22:46:12 2024 by wfish.

THIS FILE USED TO LEAK THE ANSWER

edit_files() joined season_summaries onto games on (team_id, season) - the same
season the game was played in. season_summaries holds one row per team-season,
so those are END-OF-YEAR figures. Predicting a week 3 game handed the model both
teams' full-season adjusted EPA, which includes the result of that very game and
every game after it.

It was not subtle in its effects, only in its appearance. The model beat the
market by 1.08 points in week 1, before anyone had played a snap, and its
accuracy did not improve as the season went on:

                model MAE   market MAE
    week 1          12.52        13.60      <- impossible without the answer
    weeks 2-3       12.04        11.96
    weeks 4-7       10.62        11.25
    week 12+        12.03        13.17

A real in-season model is at its worst in week 1 and improves as evidence
accumulates. Flat-and-already-good is the signature of a leak. Every MAE this
model has ever reported, and any blend weight fitted against it, was built on
that.

WHAT IT JOINS NOW

rolling_summaries.csv, on (game_id, team_id). That file carries one row per team
per game holding the rolling average of everything BEFORE it: a team's first
game of a season is empty, since nothing has happened yet, and the figures
correlate +0.26 with the margin of the game they precede - the modest number a
genuine pre-game quantity shows. A figure containing the game would correlate
far higher.

Games with no prior data - every team's season opener - are dropped rather than
filled. There is nothing to fill them with, and the blender already weights the
preseason model heavily when few games have been played.

IT ALSO GETS THE ADJUSTMENT, AS OF EACH WEEK

The rolling figures are not opponent-adjusted, so three games against weak
defences read as three good games. etl/summarize/asof_adjusted.py supplies the
missing third table: opponent-adjusted, computed only from games played before
each week. Both blocks are given to the model rather than one replacing the
other.

    2025 holdout
      rolling only                     R2 0.332   MAE 13.117
      + as-of-week adjusted            R2 0.346   MAE 12.972
      + as-of-week adjusted, gated     R2 0.351   MAE 12.929

The adjusted block is WITHHELD before week 5. A ridge fitted on two weeks of
football is mostly noise and the estimator cannot tell that from signal - given
it in weeks 2-3 it uses it and loses 0.47 of MAE there against plain rolling
form. Withholding lets it fall back on unadjusted form exactly while the
adjustment is untrustworthy, and recovers those weeks from 13.30 to 12.98. Weeks
4 through 7 were swept; 5 wins.

Gaps are left as NaN rather than filled. XGBoost learns a default direction per
split, so a missing adjustment is information rather than a hole.

IT DELIBERATELY DOES NOT SEE PRESEASON INFORMATION

Rolling form alone leaves this model with no memory: in week 2 it knows one game
about each side and nothing else. Adding last season's adjusted EPA and the
eight preseason position ratings fixes that and makes this model substantially
better ON ITS OWN, 13.49 to 12.93 MAE.

It was tried, and then removed, because it makes the SYSTEM worse. The preseason
model is built from those same quantities, so handing them here too makes the
pair redundant: the correlation between the two models' predictions rises from
+0.492 to +0.788. Blending two models that increasingly say the same thing gains
less than blending two that see different evidence.

Walked forward, blend weights fitted on 2020-23 and scored on 2024-25:

    preseason model alone                        14.098
    this model alone, WITH preseason knowledge   12.928
    this model alone, WITHOUT                    13.491
    blend, this model WITHOUT   (ships)          12.602
    blend, this model WITH                       12.743

Clean separation wins by 0.14 and wins in every week band. The early-season
weakness the preseason block was added to cure is the blender's job, and the
blender does it: weeks 2-4 come out at 13.20 blended against 14.25 for this
model alone.

The general lesson is worth keeping. A component tuned in isolation can be
genuinely improved and still leave the system worse. Changes here are judged on
the blend, not on this model's own holdout.

WHY THE SEASON RANGE STARTS AT 2017

The four skill-position ratings - quarterback, backfield, wideout, tight end -
do not exist before 2017 at all; the line, front seven, secondary and special
teams go back further. Training from 2010 meant 44% of rows carried sixteen
median-filled position columns, teaching the model that "average everything" is
the commonest roster in the country, which is an artefact rather than a fact.

Measured on the 2025 holdout, the choice barely matters:

    2010+, position filled        9,649 train   R2 0.327   MAE 13.141
    2017+, residual gaps filled   5,466 train   R2 0.332   MAE 13.117
    2017+, every value real       4,885 train   R2 0.336   MAE 13.210
    2010+, no position block      9,649 train   R2 0.311   MAE 13.262

The block is worth about 0.13 of MAE. Whether it is filled or the window is
trimmed is worth 0.02, which is noise. 2017 is used because it is the honest
window, not because it measurably wins.

Note also that train_season_range was previously accepted and ignored -
start_year and end_year were passed to edit_files and never referenced - so any
range in the experiment file was decorative. It is applied now.
"""

import os

import pandas as pd
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pickle import dump
import argparse

# Seed for the train/test split. Overridable so a change can be scored on
# several splits rather than trusting one.
SPLIT_SEED = int(os.environ.get('SPLIT_SEED', '0'))


def read_data(games_df_file_loc: str,
              season_summary_df_file_loc: str,
              # pbp_df_file_loc: str,
              experiment_info_txt_file_loc: str):

    with open(experiment_info_txt_file_loc) as f:
        lines = f.readlines()

    for line in lines:
        try:
            data = line.split(': ')[1].rstrip("\n")
        except:
            continue
        if 'train_season_range' in line:
            year_range = data
            start_year = int(year_range.split(' - ')[0])
            end_year = int(year_range.split(' - ')[1])
            continue
        elif 'data_features' in line:
            features = data.split(',')
            continue
        elif 'test_size' in line:
            test = float(data)
            continue
        elif 'test_year' in line:
            test = int(data)
        elif 'fbs_only_ind' in line:
            fbs_only_ind = bool(data)
        elif 'training_algorithm' in line:
            algo = data

    return pd.read_csv(games_df_file_loc), \
        pd.read_csv(season_summary_df_file_loc), \
        start_year, \
        end_year, \
        features, \
        test, \
        fbs_only_ind, \
        algo


def edit_files(games_df: pd.DataFrame,
               rolling_df: pd.DataFrame,
               features: list,
               start_year: int,
               end_year: int
              ):
    """Join each game to both teams' form as it stood BEFORE kickoff.

    The join key is (game_id, team_id), not (season, team_id). That is the whole
    correction: the old key matched a game to its own season's final numbers.
    """
    # The experiment file's train_season_range was accepted and then never
    # applied - start_year and end_year arrived as arguments and were unused, so
    # a file asking for 2015-2025 silently trained on 2010-2025.
    before_range = len(games_df)
    games_df = games_df[games_df['season'].between(start_year, end_year)]
    print(f"   season range {start_year}-{end_year}: "
          f"{before_range:,} games -> {len(games_df):,}")

    stats_df = rolling_df[['game_id', 'team_id'] + features]

    away_df = pd.merge(games_df, stats_df,
                       left_on=['id', 'away_team_id'],
                       right_on=['game_id', 'team_id'])
    home_df = pd.merge(away_df, stats_df,
                       left_on=['id', 'home_team_id'],
                       right_on=['game_id', 'team_id'],
                       suffixes=('_away', '_home'))

    keep_cols = []
    for i in features:
        for j in list(home_df):
            if i in j and j not in keep_cols:
                keep_cols.append(j)

    other_cols = ['season', 'id', 'date', 'short_name',
                  'away_team_id', 'home_team_id',
                  'status', 'home_score_differential']
    out = home_df[other_cols + keep_cols]
    # a season opener has no prior form for one or both sides; there is nothing
    # to fill it with, and the blender leans on the preseason model there anyway
    before = len(out)
    out = out.dropna(subset=keep_cols)
    print(f"   joined pre-game form to {before:,} games; "
          f"{before - len(out):,} dropped for having no prior data "
          f"(season openers), leaving {len(out):,}")
    # add_preseason_knowledge is deliberately NOT called - see the module
    # docstring. It improves this model and costs the blend 0.14 MAE.
    return add_asof_adjusted(out, games_df)


_HERE = os.path.dirname(os.path.abspath(__file__))
SUMMARIES = os.path.normpath(os.path.join(
    _HERE, '..', '..', 'etl', 'summarize', 'results', 'season_summaries.csv'))
POSITION_FILE = os.path.normpath(os.path.join(
    _HERE, '..', '..', 'etl', 'summarize', 'results', 'position_ratings.csv'))
ASOF_FILE = os.path.normpath(os.path.join(
    _HERE, '..', '..', 'etl', 'summarize', 'results', 'asof_adjusted.csv'))
PRIOR_FEATURES = ['adjusted_epa_per_play_off', 'adjusted_epa_per_play_def']

# The opponent adjustment is not trustworthy before this week; see the module
# docstring. Withheld rather than filled, so the model falls back on rolling.
ASOF_FROM_WEEK = 5


def _week_num(w):
    w = str(w)
    if w.strip().lower().startswith('week'):
        tail = w.split()[-1]
        if tail.isdigit():
            return int(tail)
    return 90


def add_asof_adjusted(df: pd.DataFrame, games_df: pd.DataFrame) -> pd.DataFrame:
    """Opponent-adjusted form as it stood before each game's week."""
    if not os.path.exists(ASOF_FILE):
        print(f"   {ASOF_FILE} missing; no as-of-week adjustment. Build it "
              f"with etl/summarize/asof_adjusted.py")
        return df
    wk = games_df[['id', 'week']].copy()
    wk['week_num'] = wk['week'].map(_week_num)
    df = df.merge(wk[['id', 'week_num']], on='id', how='left')

    a = pd.read_csv(ASOF_FILE, low_memory=False)
    acols = [c for c in a.columns if c.startswith('adjusted_')]
    for side in ('home', 'away'):
        r = a.rename(columns={c: f'{c}_{side}_adj' for c in acols})
        df = df.merge(r, left_on=[f'{side}_team_id', 'season', 'week_num'],
                      right_on=['team_id', 'season', 'week_num'], how='left')
        df = df.drop(columns=['team_id'])
    cols = [c for c in df.columns if c.endswith('_adj')]
    df.loc[df['week_num'] < ASOF_FROM_WEEK, cols] = np.nan
    have = df[cols].notna().all(axis=1).mean()
    print(f"   as-of-week adjustment on {have:.1%} of games "
          f"(withheld before week {ASOF_FROM_WEEK}); gaps left as NaN")
    return df


def add_preseason_knowledge(df: pd.DataFrame) -> pd.DataFrame:
    """Attach last season's adjusted EPA and this season's position ratings.

    Both are lagged or preseason-stamped already, so neither can see the game.
    Prior EPA is season S-1's, joined onto season S. The position ratings are
    built by their own modules to stand before the season they name and are
    checked for that by position_ratings.py --check.
    """
    if os.path.exists(SUMMARIES):
        s = pd.read_csv(SUMMARIES, low_memory=False)
        have = [c for c in PRIOR_FEATURES if c in s.columns]
        prior = s[['team_id', 'season'] + have].copy()
        prior['season'] += 1          # last season's figure, this season's row
        df = _attach(df, prior, have, 'pri')
    else:
        print(f"   {SUMMARIES} missing; no prior-season EPA")

    if os.path.exists(POSITION_FILE):
        p = pd.read_csv(POSITION_FILE, low_memory=False)
        pf = [c for c in p.columns if c.startswith('pf_')]
        df = _attach(df, p[['team_id', 'season'] + pf], pf, 'pos')
        cols = [c for c in df.columns if c.endswith('_pos')]
        cov = df[cols].notna().all(axis=1).mean() if cols else 0
        # filled rather than dropped: these start in 2017 and need a roster
        # that links to recruiting records, so about half the rows lack them
        df[cols] = df[cols].fillna(df[cols].median())
        print(f"   position ratings on {cov:.1%} of games, "
              f"rest filled at the median")
    else:
        print(f"   {POSITION_FILE} missing; no position ratings")
    return df


def _attach(df, src, cols, tag):
    for side in ('home', 'away'):
        s = src.rename(columns={c: f'{c}_{side}_{tag}' for c in cols})
        df = df.merge(s, left_on=[f'{side}_team_id', 'season'],
                      right_on=['team_id', 'season'], how='left')
        df = df.drop(columns=['team_id'])
    return df


def split_data(data,
               test,
               y_col):
    """

    :param data: (pd.DataFrame) dataframe to be split into train and test dfs
    :param test_size: (float) percentage of dataset to allocate to test
    :param y_col: (str) name of column containing y data
    :return: train_data, test_data
    """

    seasons = list(data.season.unique())
    # if test is a float that means that the input mustve been a percentage to holdout as test
    if isinstance(test, float) is True:

        train_dfs = []
        test_dfs = []
        for season in seasons:
            season_data = data.loc[data['season'] == season]
            X = season_data.drop(y_col, axis=1)
            y = season_data[y_col]

            # Seeded, as in the preseason model. train_model.py fixes the
            # ESTIMATOR's random_state but the split had none, so every run
            # scored a different test set and two runs were never comparable.
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test, random_state=SPLIT_SEED)
            train_dfs.append(pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train)], axis=1))
            test_dfs.append(pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_test)], axis=1))

        # 'status' was dropped in the year branch below but not here, so it
        # survived as a feature on one path and not the other
        drop = [c for c in ('season', 'status') if c in train_dfs[0].columns]
        return pd.concat(train_dfs).drop(drop, axis=1), \
            pd.concat(test_dfs).drop(drop, axis=1)

    # if test is an int that means that the input mustve been a year to holdout as test
    elif isinstance(test, int) is True:

        return data.loc[data['season'] != test].drop(['season', 'status'], axis=1), \
            data.loc[data['season'] == test].drop(['season', 'status'], axis=1)


def final_edits(train_df: pd.DataFrame,
                test_df: pd.DataFrame,
                features: list,
                y_col: str):

    identifier_cols = ['id', 'date', 'short_name', 'away_team_id', 'home_team_id']
    scaler = StandardScaler()  # or: MinMaxScaler(feature_range=(0,1))

    train = train_df[[y_col] + features]
    train_id_df = train_df[identifier_cols]
    test = test_df[[y_col] + features]
    test_id_df = test_df[identifier_cols]

    train_X = train.drop(y_col, axis=1)
    train_y = train[y_col]

    train_X_scaled = scaler.fit_transform(train_X)

    test_X = test.drop(y_col, axis=1)
    test_y = test[y_col]

    test_X_scaled = scaler.transform(test_X)

    # save the scaler
    dump(scaler, open('temp/scaler.pkl', 'wb'))

    return pd.DataFrame(train_X_scaled, columns=features), \
        pd.DataFrame(test_X_scaled, columns=features), \
        train_y, \
        test_y, \
        train_id_df, \
        test_id_df


#def parse_args():

#    parser = argparse.ArgumentParser(description='Input path for season summary csv file')
#    parser.add_argument('--season_summary_data')
#    args = parser.parse_args()

#    return args.season_summary_data


if __name__ == '__main__':

    # All files ending with .txt
    experiment_info_txt_file_loc = glob.glob('temp/in_season_experiment*')[0]
    #season_summary_data = parse_args()

    games_df, \
        season_summaries_df, \
        start_year, \
        end_year, \
        features, \
        test,\
        fbs_only_ind, \
        algo = read_data(games_df_file_loc='temp/games.csv',
            season_summary_df_file_loc='temp/rolling_summaries.csv',
            experiment_info_txt_file_loc=experiment_info_txt_file_loc)

    # named season_summaries_df by history; it now holds ROLLING summaries,
    # one row per team per game, carrying only what preceded that game
    season_summaries_df.to_csv('temp/rolling_summaries_raw.csv')

    train_df = edit_files(games_df=games_df,
                          rolling_df=season_summaries_df,
                          features=features,
                          start_year=start_year,
                          end_year=end_year)

    train_df.to_csv('temp/raw_train.csv')

    train_df, \
        test_df = split_data(data=train_df,
                             test=test,
                             y_col='home_score_differential')

    train_df.to_csv('temp/train_df.csv')
    test_df.to_csv('temp/test_df.csv')
    
    final_cols = []
    for i in features:
        final_cols.append(i + '_home')
        final_cols.append(i + '_away')
    # the preseason and as-of-week blocks are named <col>_<side>_pri / _pos /
    # _adj rather than <col>_<side>, so they are picked up separately. Order
    # matters: the scaler and estimator index by position, not by name.
    extra = sorted(c for c in train_df.columns
                   if c.endswith('_pri') or c.endswith('_pos')
                   or c.endswith('_adj'))
    final_cols += extra
    n_adj = sum(c.endswith('_adj') for c in extra)
    print(f"   {len(final_cols)} features: {len(final_cols) - len(extra)} "
          f"rolling, {len(extra) - n_adj} preseason, {n_adj} as-of-week")

    train_X_scaled, \
        test_X_scaled, \
        train_y, \
        test_y,\
        train_id_df,\
        test_id_df = final_edits(train_df=train_df,
                                 test_df=test_df,
                                 features=final_cols,
                                 y_col='home_score_differential')

    train_X_scaled.to_csv('temp/train_X_scaled.csv')
    test_X_scaled.to_csv('temp/test_X_scaled.csv')
    pd.DataFrame(train_y, columns=['home_score_differential']).to_csv('temp/train_y.csv')
    pd.DataFrame(test_y, columns=['home_score_differential']).to_csv('temp/test_y.csv')
    train_id_df.to_csv('temp/train_id_cols.csv')
    test_id_df.to_csv('temp/test_id_cols.csv')
