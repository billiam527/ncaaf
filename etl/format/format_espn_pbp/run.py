import pandas as pd
from edit_pbp_file import add_binary_play_stats, \
    generate_cum_stats, \
    calc_new_features
import argparse
import datetime

parser = argparse.ArgumentParser(description='Input path for play by play csv file')
default_input_file_path = 'temp/pbp.csv'
parser.add_argument('--input_file_path', default=default_input_file_path, help='File path for pbp csv file')
default_output_file_path = 'temp/pbp_edit.csv'
parser.add_argument('--output_file_path', default=default_output_file_path, help='File path for pbp csv file')
today = (datetime.datetime.today()).strftime('%Y-%m-%d')
args = parser.parse_args()
input_file_path = args.input_file_path
output_file_path = args.output_file_path


def main(input_file_path=input_file_path,
         output_file_path=output_file_path):

    df = pd.read_csv(input_file_path)
    if df.empty is False:
        df = add_binary_play_stats(df)
        df = generate_cum_stats(df)
        df = calc_new_features(df)
        df.set_index('id', inplace=True)
        df.to_csv(output_file_path)


if __name__ == '__main__':
   main()
