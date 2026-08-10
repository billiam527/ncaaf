#!/usr/bin/env bash
set -e

# Pull the inputs preprocess.py reads. season_summaries.csv is produced by the
# ETL (etl/summarize) and published to model_data/ by etl.sh.
aws s3 cp s3://ncaaf-data/espn-games-data/games/csvs/games.csv temp/games.csv
aws s3 cp s3://ncaaf-data/model_data/season_summaries.csv temp/season_summaries.csv

file=$(python -c "import glob; print(glob.glob('temp/in_season_experiment*')[0])")

FBS_ind=""
SUB='fbs_only_ind'
while read p; do
if [[ "$p" == *"$SUB"* ]]; then
  FBS_ind=$(echo "$p"|awk -F ": " '{print $NF}')
fi
done < "$file"

#### if fbs ind = True, join games to teams/fbs_ind and drop any game
#### containing a team that isn't in the FBS list
if [[ "$FBS_ind" == "True" ]]; then

  aws s3 cp s3://ncaaf-data/espn-teams-data/teams.csv temp/teams.csv

  python - <<'PY'
import pandas as pd
games = pd.read_csv('temp/games.csv', low_memory=False)
teams = pd.read_csv('temp/teams.csv')
teams = teams.loc[teams['fbs_ind'] == 1.0][['id', 'fbs_ind']]
teams = teams.rename(columns={'id': 'team_id'})
merged = games.merge(teams, left_on='home_team_id', right_on='team_id')
merged = merged.merge(teams, left_on='away_team_id', right_on='team_id')
merged.to_csv('temp/games.csv')
print(f"   FBS filter: {len(games)} -> {len(merged)} games")
PY

fi

# NOTE: this used to download pbp_edit.csv (~1.5 GB), filter it, and re-run
# etl/summarize/summarize_games.py to rebuild FBS-only summaries. That call
# passed --games_file_loc / --pbp_file_loc / --summary_stats, none of which are
# real arguments, so it failed on every run; with no `set -e` the failure was
# swallowed and training silently used the season_summaries.csv downloaded
# above. It also wrote to etl/summarize/results, so had it ever worked it would
# have overwritten the ETL's own outputs. Removed rather than repaired: the ETL
# already produces these summaries, and preprocess.py applies the FBS filter
# through the games join above.

python preprocess.py
