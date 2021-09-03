pyinstaller --onefile world_vaccination_predictor.py
mv ./dist/world_vaccination_predictor ./
rm -rf dist build __pycache__ world_vaccination_predictor.spec
