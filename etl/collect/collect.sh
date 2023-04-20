#!/usr/bin/env bash

cd collect_espn_games
./collect_espn_games.sh 2022 2023
cd ..

cd collect_espn_pbp
./collect_espn_pbp.sh 2022 2023
cd ..

cd collect_espn_teams
./collect_ESPN_teams.sh
cd ..