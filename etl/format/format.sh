#!/usr/bin/env bash

cd format_espn_games
./format_espn_games.sh 2010 2022
cd ..

cd format_espn_pbp
./format_espn_pbp.sh 2010 2022
cd ..