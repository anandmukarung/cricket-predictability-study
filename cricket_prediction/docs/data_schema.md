# Training Table Draft Schema

One row per match.

## Core columns
- `match_id`
- `date`
- `format`
- `competition`
- `team_1`
- `team_2`
- `winner`
- `label_team_1_win`

## Team-history feature placeholders
- `team_1_win_pct_last_5`
- `team_2_win_pct_last_5`
- `team_1_avg_runs_last_5`
- `team_2_avg_runs_last_5`
- `team_1_avg_runs_conceded_last_5`
- `team_2_avg_runs_conceded_last_5`

## Player aggregate feature placeholders
- `team_1_batting_quality`
- `team_2_batting_quality`
- `team_1_bowling_quality`
- `team_2_bowling_quality`
- `team_1_experience_score`
- `team_2_experience_score`

## Notes
- No post-match fields.
- Toss should only be included if prediction time is explicitly set to post-toss.
- Keep raw names and normalized IDs where possible.
