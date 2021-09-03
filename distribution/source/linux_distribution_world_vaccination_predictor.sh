#!/bin/bash
git init
git pull https://github.com/owid/covid-19-data.git
git restore public
cp ./public/data/vaccinations/vaccinations.csv ../resource/
rm -rf public
rm -rf scripts
rm -f README.md
./world_vaccination_predictor