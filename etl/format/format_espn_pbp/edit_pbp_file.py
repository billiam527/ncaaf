import pandas as pd


def add_binary_play_stats(df: pd.DataFrame) -> pd.DataFrame:

    """
    Create desired passing statistics to the dataframe

    param df: (pd.DataFrame) Dataframe of play by play df

    return df: (pd.DataFrame) play by play DataFrame with binary statistics
    """

    play_id_list = [3,  # Pass incompletion
                    4,  # Pass completion
                    5,  # Rush
                    6,  # Pass interception
                    7,  # Sack
                    8,  # Penalty
                    9,  # Fumble recovery (own)
                    24,  # Pass reception
                    26,  # Pass interception return
                    29,  # Fumble recovery (opp)
                    36,  # Interception return (TD)
                    51,  # Pass
                    63,  # Interception
                    67,  # Pass TD
                    68  # Rush TD
                    ]

    # offensive play count
    df.loc[df['play_type_id'].isin(play_id_list), 'offensive_play'] = 1
    df['offensive_play'] = df['offensive_play'].fillna(0)
    df['offensive_play'] = df['offensive_play'].astype('int')

    # 1st down play success
    df.loc[(df['play_type_id'].isin(play_id_list)) & (df['down'] == 1) &
           ((df['stat_yardage'] / df['distance']) >= .5), 'successful_play'] = 1

    # 2nd down play success
    df.loc[(df['play_type_id'].isin(play_id_list)) & (df['down'] == 2) &
           ((df['stat_yardage'] / df['distance']) >= .75), 'successful_play'] = 1

    # 3rd and 4th down play success
    df.loc[(df['play_type_id'].isin(play_id_list)) & (df['down'] >= 3) &
           ((df['stat_yardage'] / df['distance']) >= 1), 'successful_play'] = 1

    # non play success
    df['successful_play'] = df['successful_play'].fillna(0)
    df['successful_play'] = df['successful_play'].astype('int')

    # offensive yards
    df.loc[df['play_type_id'].isin(play_id_list), 'offensive_yards'] = df['stat_yardage']
    df['offensive_yards'] = df['offensive_yards'].fillna(0)

    return df


def generate_cum_stats(df):

    """
    Aggregate the passing statistic by game/team

    param df: (pd.DataFrame) play by play DataFrame with binary statistics

    return df: (pd.DataFrame) play by play DataFrame with cumulative data
    """

    # Cum totals
    df['cum_plays'] = df.groupby(['game_id', 'team_id'])['offensive_play'].cumsum()
    df['cum_plays'] = df['cum_plays'].astype('int')

    df['cum_yards'] = df.groupby(by=['game_id', 'team_id'])['offensive_yards'].cumsum()

    df['cum_successful_plays'] = df.groupby(by=['game_id', 'team_id'])['successful_play'].cumsum()
    df['cum_successful_plays'] = df['cum_successful_plays'].astype('int')

    return df


def calc_new_features(df):

    """
    Calculate desired features

    param df: (pd.DataFrame) cumulative DataFrame

    return df: (pd.DataFrame) DataFrame with calculated data
    """

    # yards per play
    df['yards_per_play'] = (df['cum_yards']/df['cum_plays']).round(2)
    df['yards_per_play'].fillna(0, inplace=True)

    # play success
    df['play_success'] = (df['cum_successful_plays']/df['cum_plays']).round(2)
    df['play_success'].fillna(0, inplace=True)

    return df


if __name__ == '__main__':
    None
