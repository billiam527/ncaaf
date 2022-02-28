import pandas as pd
import argparse

parser = argparse. \
    ArgumentParser(description='Input game dates (oldest to newest) to pull NCAA Football calendar weeks')

parser.add_argument('--file1', help='Dataframe 1 to merge (file path)')
parser.add_argument('--file2', help='Dataframe 2 to merge (file path)')
parser.add_argument('--on', help='Column to merge on')
parser.add_argument('--how', help='How to merge columns (inner, outer, left, right)')
parser.add_argument('--fillna', help='Dataframe 1 to merge')

args = parser.parse_args()
file1 = args.file1
file2 = args.file2
on = args.on
how = args.how
fillna = args.fillna


def read_in_data(file1: str,
                 file2: str):

    """

    param file1: (str) file location of csv file to read in
    param file2: (str) file location of csv file to read in

    return:
    """

    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    return df1, df2


def merge(df1: pd.DataFrame,
          df2: pd.DataFrame,
          on: str,
          how: str,
          fillna):

    """

    param df1: (pd.DataFrame) pandas dataframe 1 to merge
    param df2: (pd.DataFrame) pandas dataframe 2 to merge
    param on: (str) how to merge dataframes (inner, outer, left, right)
    param how: (str) column to merge on
    param fillna: column to fill missing values with

    return: (pd.DataFrame)
    """

    df1[on] = df1[on].str.split('T').str[0]
    merged_df = pd.merge(df1, df2, on=on, how=how).fillna(fillna)

    return merged_df


def create_score_differentials_and_total(df: pd.DataFrame):

    df['home_score_differential'] = df['home_score'] - df['away_score']
    df['total'] = df['home_score'] + df['away_score']
    df['home_first_quarter_score_differential'] = df['home_first_quarter'] - df['away_first_quarter']
    df['first_quarter_total'] = df['home_first_quarter'] + df['away_first_quarter']
    df['home_first_half_score_differential'] = df['home_first_quarter'] + df['home_second_quarter'] \
        - df['away_first_quarter'] - df['away_second_quarter']
    df['first_half_total'] = df['home_first_quarter'] + df['home_second_quarter'] \
        + df['away_first_quarter'] + df['away_second_quarter']

    return df


if __name__ == '__main__':

    df1, df2 = read_in_data(file1, file2)
    df = merge(df1, df2, on, how, fillna)
    df = create_score_differentials_and_total(df)
    df.to_csv('temp/new_games.csv', index=False)
