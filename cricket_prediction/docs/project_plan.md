# Project Plan

## Working Title
T20 Match Outcome Prediction Using Team History and Aggregated Player-Quality Features

## Research Questions
1. How well can classical pattern recognition methods predict T20 match outcomes using only pre-match information?
2. How much predictive value comes from player-quality-based aggregate features beyond simpler team-history features?

## Initial Scope
- Start with T20 match outcome prediction
- Use Cricsheet as the core match data source
- Keep prediction strictly pre-match to avoid leakage
- Aggregate player-level information to team-level features
- Compare baseline, logistic regression, LDA, and SVM

## Stretch Goal
Extend the same framework to ODI and Test formats to compare predictability and variability across cricket formats.

## First Milestone
Define and build a match-level training table with:
- one row per match
- binary win/loss label
- only pre-match features
- clean team and player identifiers

## Risks
- player/team name mismatches
- lineup availability
- data leakage from post-match fields
- too many feature ideas too early

## Task Split (Draft)
### Anand
- dataset scope and filtering rules
- Cricsheet ingestion decisions
- player-quality feature philosophy
- feature design and analysis framing

### Abdullah
- modeling pipeline setup
- baseline/logistic/LDA/SVM experiment scaffolding
- metric tables and reproducibility support

### Shared
- validation design
- report writing
- final presentation
