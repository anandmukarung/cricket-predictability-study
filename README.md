# T20 Match Prediction

This project studies T20 cricket match outcome prediction using pre-match structured features derived from historical team performance, player-quality proxies, and contextual information.

## Initial Scope
- Data source: Cricsheet
- Prediction task: pre-match win/loss prediction for T20 matches
- Modeling: baseline, logistic regression, LDA, SVM
- Longer-term goal: analyze how player-quality aggregates contribute to team-level prediction and what model performance suggests about the variability of T20 cricket

## Repository Structure
- `data/raw/` raw source files and external reference data
- `data/interim/` partially cleaned tables
- `data/processed/` modeling-ready tables
- `docs/` project planning and proposal notes
- `notebooks/` exploratory analysis and experiment notebooks
- `src/` reusable Python code for ingestion, preprocessing, features, and models
- `reports/figures/` exported plots and figures
- `tests/` lightweight sanity checks

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## First Milestone
Create one processed match-level training table with one row per match and only pre-match features.
