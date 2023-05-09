import pandas as pd
import argparse


def read_and_edit_files(pbp_file_loc: str,
                        games_file_loc: str,
                        summary_stats: list):

    # read in data
    pbp = pd.read_csv(pbp_file_loc)
    games = pd.read_csv(games_file_loc)

    # make small edits
    games_edit = games[['id', 'date', 'season', 'home_team_id', 'away_team_id', 'week']]
    pbp_list = ['id', 'game_id', 'team_id', 'offensive_play'] + summary_stats
    pbp_edit = pbp[pbp_list]
    games_edit['id'] = pd.to_numeric(games_edit['id'])
    pbp_edit['game_id'] = pd.to_numeric(pbp_edit['game_id'])
    pbp_edit['play_id'] = pbp_edit['id']
    pbp_edit = pbp_edit.drop('id', axis=1)

    # merge game and pbp df
    return pd.merge(games_edit, pbp_edit, left_on='id', right_on='game_id')


def summarize_games(df: pd.DataFrame,
                    statistics: list):

    # home offensive summaries
    home_o = df.loc[(df['team_id'] == df['home_team_id']) & (df['offensive_play'] == 1)]
    home_o_game = home_o[home_o.groupby('game_id')['play_id'].transform('max') == home_o['play_id']] \
        .groupby(['id', 'date', 'season']) \
        .sum()[['home_team_id'] + statistics] \
        .rename({'home_team_id': 'team_id'}, axis=1)

    # home defensive statistics
    home_d = df.loc[(df['team_id'] == df['away_team_id']) & (df['offensive_play'] == 1)]
    home_d_game = home_d[home_d.groupby('game_id')['play_id'].transform('max') == home_d['play_id']] \
        .groupby(['id', 'date', 'season']) \
        .sum()[['home_team_id'] + statistics] \
        .rename({'home_team_id': 'team_id'}, axis=1)

    # away offensive summaries
    away_o = df.loc[(df['team_id'] == df['away_team_id']) & (df['offensive_play'] == 1)]
    away_o_game = away_o[away_o.groupby('game_id')['play_id'].transform('max') == away_o['play_id']] \
        .groupby(['id', 'date', 'season']) \
        .sum()[['away_team_id'] + statistics] \
        .rename({'away_team_id': 'team_id'}, axis=1)

    # away defensive statistics
    away_d = df.loc[(df['team_id'] == df['home_team_id']) & (df['offensive_play'] == 1)]
    away_d_game = away_d[away_d.groupby('game_id')['play_id'].transform('max') == away_d['play_id']] \
        .groupby(['id', 'date', 'season']) \
        .sum()[['away_team_id'] + statistics] \
        .rename({'away_team_id': 'team_id'}, axis=1)

    return home_o_game, home_d_game, away_o_game, away_d_game


def season_summaries(home_o_games: pd.DataFrame,
                     home_d_games: pd.DataFrame,
                     away_o_games: pd.DataFrame,
                     away_d_games: pd.DataFrame,
                     statistics: list):

    # season off medians
    season_off_medians = pd.concat([home_o_games.reset_index(), away_o_games.reset_index()]) \
        .groupby(['team_id', 'season']) \
        .median()[statistics]

    # season def medians
    season_def_medians = pd.concat([home_d_games.reset_index(), away_d_games.reset_index()]) \
        .groupby(['team_id', 'season']) \
        .median()[statistics]

    # join off and def dfs
    return pd.merge(season_off_medians,
        season_def_medians,
        right_index=True,
        left_index=True,
        suffixes=['_off', '_def'])


def game_summaries(home_o_games: pd.DataFrame,
                   home_d_games: pd.DataFrame,
                   away_o_games: pd.DataFrame,
                   away_d_games: pd.DataFrame):

    # home game by game medians
    home_gbg = pd.merge(home_o_games, home_d_games,
                        right_index=True, left_index=True,
                        suffixes=['_off', '_def']) \
        .drop(['team_id_def'], axis=1) \
        .rename({'team_id_off': 'team_id'}, axis=1) \
        .reset_index() \
        .set_index(['season', 'date', 'id', 'team_id']) \
        .sort_index(level=['season', 'team_id', 'date', 'id'])

    # away game by game medians
    away_gbg = pd.merge(away_o_games, away_d_games,
                        right_index=True, left_index=True,
                        suffixes=['_off', '_def']) \
        .drop(['team_id_def'], axis=1) \
        .rename({'team_id_off': 'team_id'}, axis=1) \
        .reset_index() \
        .set_index(['season', 'date', 'id', 'team_id']) \
        .sort_index(level=['season', 'team_id', 'date', 'id'])

    # game by game medians
    return pd.concat([home_gbg, away_gbg]) \
        .reset_index() \
        .set_index(['season', 'date', 'id', 'team_id']) \
        .sort_index(level=['season', 'team_id', 'date', 'id'])


def rolling_game_summaries(gbg_df: pd.DataFrame,
                           stats: list):

    # game by game rolling averages
    gbg_reset = gbg_df.reset_index()

    # go through statistics list and get the rolling median of each
    group_by = ['season', 'team_id']
    statistics = []
    for stat in stats:
        statistics.append(stat + '_off')
        statistics.append(stat + '_def')

    moving_statistics = []
    for stat in statistics:
        gbg_reset['rolling_' + stat] = gbg_reset.groupby(group_by)[stat].transform(lambda x: x.expanding(1).median())
        moving_statistics.append('rolling_' + stat)

    # Shift down by one game. We would not know the statistics of the first game prior to it being played
    return gbg_reset.set_index(['season', 'team_id', 'date', 'id']) \
        .groupby(group_by)[moving_statistics] \
        .shift(1)


def parse_args():

    parser = argparse.ArgumentParser(description='Input path for play by play csv file')
    default_games_file_path = 'temp/games.csv'
    parser.add_argument('--games_file_loc', default=default_games_file_path)
    default_pbp_file_path = 'temp/pbp_edit.csv'
    parser.add_argument('--pbp_file_loc', default=default_pbp_file_path)
    default_output_file_path = 'temp/season_summaries.csv'
    parser.add_argument('--output_file_loc', default=default_output_file_path)
    parser.add_argument('--summary_stats', nargs="+")
    args = parser.parse_args()

    return args.games_file_loc, \
        args.pbp_file_loc, \
        args.output_file_loc, \
        args.summary_stats


if __name__ == '__main__':

    games_file_loc, pbp_file_loc, output_file_loc, summary_stats = parse_args()

    games_pbp_df = read_and_edit_files(pbp_file_loc=pbp_file_loc,
        games_file_loc=games_file_loc,
        summary_stats=summary_stats)

    home_o_games, \
        home_d_games, \
        away_o_games, \
        away_d_games = summarize_games(df=games_pbp_df, statistics=summary_stats)

    season_df = season_summaries(home_o_games=home_o_games,
        home_d_games=home_d_games,
        away_o_games=away_o_games,
        away_d_games=away_d_games,
        statistics=summary_stats)

    game_by_game_summaries = game_summaries(home_o_games=home_o_games,
        home_d_games=home_d_games,
        away_o_games=away_o_games,
        away_d_games=away_d_games)

    rolling_summaries = rolling_game_summaries(gbg_df=game_by_game_summaries, stats=summary_stats)

    season_df.to_csv(output_file_loc + 'season_summaries.csv')
    game_by_game_summaries.to_csv('temp/game_by_game_summaries.csv')
    rolling_summaries.to_csv('temp/rolling_summaries.csv')
