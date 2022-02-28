#!/usr/bin/env bash

cd collect_espn_games
./collect_espn_games.sh 2020 2021
cd ..

cd collect_espn_pbp
./collect_espn_pbp.sh 2020 2021
cd ..