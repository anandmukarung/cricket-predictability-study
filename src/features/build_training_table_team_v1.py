from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import pandas as pd


MATCHES_MASTER_PATH = Path("data/interim/matches_master.parquet")
TEAM_MATCH_STATS_PATH = Path("data/interim/team_match_stats.parquet")

OUTPUT_DIR = Path("data/processed")
OUTPUT_CSV = OUTPUT_DIR / "training_table_team_v1.csv"
OUTPUT_PARQUET = OUTPUT_DIR / "training_table_team_v1.parquet"


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def make_empty_team_state() -> dict[str, Any]:
    return {
        "matches": 0,
        "wins": 0,
        "runs_scored": 0,
        "runs_conceded": 0,
        "wickets_lost": 0,
        "wickets_taken": 0,
        "balls_faced": 0,
        "balls_bowled": 0,
        "last5_wins": deque(maxlen=5),
        "last5_runs_scored": deque(maxlen=5),
        "last5_runs_conceded": deque(maxlen=5),
        "last5_wickets_lost": deque(maxlen=5),
        "last5_wickets_taken": deque(maxlen=5),
    }


def summarize_team_state(state: dict[str, Any], prefix: str) -> dict[str, Any]:
    matches = state["matches"]
    wins = state["wins"]
    runs_scored = state["runs_scored"]
    runs_conceded = state["runs_conceded"]
    wickets_lost = state["wickets_lost"]
    wickets_taken = state["wickets_taken"]
    balls_faced = state["balls_faced"]
    balls_bowled = state["balls_bowled"]

    last5_wins = list(state["last5_wins"])
    last5_runs_scored = list(state["last5_runs_scored"])
    last5_runs_conceded = list(state["last5_runs_conceded"])
    last5_wickets_lost = list(state["last5_wickets_lost"])
    last5_wickets_taken = list(state["last5_wickets_taken"])

    return {
        f"{prefix}_prior_matches": matches,
        f"{prefix}_prior_win_pct": safe_div(wins, matches),
        f"{prefix}_prior_avg_runs_scored": safe_div(runs_scored, matches),
        f"{prefix}_prior_avg_runs_conceded": safe_div(runs_conceded, matches),
        f"{prefix}_prior_avg_wickets_lost": safe_div(wickets_lost, matches),
        f"{prefix}_prior_avg_wickets_taken": safe_div(wickets_taken, matches),
        f"{prefix}_prior_run_rate": safe_div(runs_scored * 6.0, balls_faced) if balls_faced else 0.0,
        f"{prefix}_prior_conceded_run_rate": safe_div(runs_conceded * 6.0, balls_bowled) if balls_bowled else 0.0,
        f"{prefix}_last5_matches": len(last5_wins),
        f"{prefix}_last5_win_pct": safe_div(sum(last5_wins), len(last5_wins)),
        f"{prefix}_last5_avg_runs_scored": safe_div(sum(last5_runs_scored), len(last5_runs_scored)),
        f"{prefix}_last5_avg_runs_conceded": safe_div(sum(last5_runs_conceded), len(last5_runs_conceded)),
        f"{prefix}_last5_avg_wickets_lost": safe_div(sum(last5_wickets_lost), len(last5_wickets_lost)),
        f"{prefix}_last5_avg_wickets_taken": safe_div(sum(last5_wickets_taken), len(last5_wickets_taken)),
    }


def update_team_state(state: dict[str, Any], team_row: pd.Series) -> None:
    won = int(bool(team_row["won"]))
    runs_scored = int(team_row["runs_scored"])
    runs_conceded = int(team_row["runs_conceded"])
    wickets_lost = int(team_row["wickets_lost"])
    wickets_taken = int(team_row["wickets_taken"])
    balls_faced = int(team_row["legal_balls_faced"])
    balls_bowled = int(team_row["legal_balls_bowled"])

    state["matches"] += 1
    state["wins"] += won
    state["runs_scored"] += runs_scored
    state["runs_conceded"] += runs_conceded
    state["wickets_lost"] += wickets_lost
    state["wickets_taken"] += wickets_taken
    state["balls_faced"] += balls_faced
    state["balls_bowled"] += balls_bowled

    state["last5_wins"].append(won)
    state["last5_runs_scored"].append(runs_scored)
    state["last5_runs_conceded"].append(runs_conceded)
    state["last5_wickets_lost"].append(wickets_lost)
    state["last5_wickets_taken"].append(wickets_taken)


def build_training_table(matches_master_df: pd.DataFrame, team_match_stats_df: pd.DataFrame) -> pd.DataFrame:
    matches_master_df = matches_master_df.sort_values(
        ["date_start", "source_bucket", "competition_folder", "file_name"],
        na_position="last"
    ).reset_index(drop=True)

    team_match_stats_df = team_match_stats_df.sort_values(
        ["date_start", "match_id", "team"],
        na_position="last"
    ).reset_index(drop=True)

    team_rows_by_match: dict[str, pd.DataFrame] = {
        match_id: grp.reset_index(drop=True)
        for match_id, grp in team_match_stats_df.groupby("match_id", sort=False)
    }

    team_state: dict[str, dict[str, Any]] = defaultdict(make_empty_team_state)
    training_rows: list[dict[str, Any]] = []

    for i, match_row in enumerate(matches_master_df.itertuples(index=False), start=1):
        match = match_row._asdict()
        match_id = match["match_id"]
        team1 = match.get("team1")
        team2 = match.get("team2")

        if not team1 or not team2:
            continue

        team1_state = team_state[team1]
        team2_state = team_state[team2]

        row = {
            "match_id": match_id,
            "date_start": match.get("date_start"),
            "source_bucket": match.get("source_bucket"),
            "official_status": match.get("official_status"),
            "competition_folder": match.get("competition_folder"),
            "event_name": match.get("event_name"),
            "match_type": match.get("match_type"),
            "gender": match.get("gender"),
            "team1": team1,
            "team2": team2,
            "venue": match.get("venue"),
            "city": match.get("city"),
            "toss_winner": match.get("toss_winner"),
            "toss_decision": match.get("toss_decision"),
            "winner": match.get("winner"),
            "label_team1_win": int(match.get("winner") == team1) if pd.notna(match.get("winner")) else None,
            "has_winner": bool(match.get("has_winner")) if "has_winner" in match else pd.notna(match.get("winner")),
            "is_binary_outcome": bool(match.get("is_binary_outcome")) if "is_binary_outcome" in match else (
                match.get("winner") in {team1, team2}
            ),
            "toss_winner_is_team1": int(match.get("toss_winner") == team1) if pd.notna(match.get("toss_winner")) else 0,
            "toss_winner_is_team2": int(match.get("toss_winner") == team2) if pd.notna(match.get("toss_winner")) else 0,
            "team1_won_toss_and_batted": int(
                match.get("toss_winner") == team1 and match.get("toss_decision") == "bat"
            ),
            "team1_won_toss_and_fielded": int(
                match.get("toss_winner") == team1 and match.get("toss_decision") == "field"
            ),
            "team2_won_toss_and_batted": int(
                match.get("toss_winner") == team2 and match.get("toss_decision") == "bat"
            ),
            "team2_won_toss_and_fielded": int(
                match.get("toss_winner") == team2 and match.get("toss_decision") == "field"
            ),
        }

        row.update(summarize_team_state(team1_state, "team1"))
        row.update(summarize_team_state(team2_state, "team2"))

        # Simple difference features
        row["diff_prior_matches"] = row["team1_prior_matches"] - row["team2_prior_matches"]
        row["diff_prior_win_pct"] = row["team1_prior_win_pct"] - row["team2_prior_win_pct"]
        row["diff_prior_avg_runs_scored"] = row["team1_prior_avg_runs_scored"] - row["team2_prior_avg_runs_scored"]
        row["diff_prior_avg_runs_conceded"] = row["team1_prior_avg_runs_conceded"] - row["team2_prior_avg_runs_conceded"]
        row["diff_prior_run_rate"] = row["team1_prior_run_rate"] - row["team2_prior_run_rate"]
        row["diff_prior_conceded_run_rate"] = row["team1_prior_conceded_run_rate"] - row["team2_prior_conceded_run_rate"]
        row["diff_last5_win_pct"] = row["team1_last5_win_pct"] - row["team2_last5_win_pct"]
        row["diff_last5_avg_runs_scored"] = row["team1_last5_avg_runs_scored"] - row["team2_last5_avg_runs_scored"]
        row["diff_last5_avg_runs_conceded"] = row["team1_last5_avg_runs_conceded"] - row["team2_last5_avg_runs_conceded"]

        training_rows.append(row)

        # Update state only after features are written, to avoid leakage
        match_team_rows = team_rows_by_match.get(match_id)
        if match_team_rows is None or match_team_rows.empty:
            continue

        for _, team_row in match_team_rows.iterrows():
            update_team_state(team_state[team_row["team"]], team_row)

        if i % 250 == 0:
            print(f"Processed {i} matches into training rows...")

    training_df = pd.DataFrame(training_rows)
    if "date_start" in training_df.columns:
        training_df["date_start"] = pd.to_datetime(training_df["date_start"], errors="coerce")

    training_df = training_df.sort_values(
        ["date_start", "source_bucket", "competition_folder", "match_id"],
        na_position="last"
    ).reset_index(drop=True)

    return training_df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MATCHES_MASTER_PATH.exists():
        raise FileNotFoundError(f"{MATCHES_MASTER_PATH} not found. Build matches_master first.")
    if not TEAM_MATCH_STATS_PATH.exists():
        raise FileNotFoundError(f"{TEAM_MATCH_STATS_PATH} not found. Build team_match_stats first.")

    matches_master_df = pd.read_parquet(MATCHES_MASTER_PATH)
    team_match_stats_df = pd.read_parquet(TEAM_MATCH_STATS_PATH)

    print(f"Loaded matches_master: {len(matches_master_df)} rows")
    print(f"Loaded team_match_stats: {len(team_match_stats_df)} rows")

    training_df = build_training_table(matches_master_df, team_match_stats_df)

    training_df.to_csv(OUTPUT_CSV, index=False)
    training_df.to_parquet(OUTPUT_PARQUET, index=False)

    print("Done. Wrote:")
    print(f"  - {OUTPUT_CSV}")
    print(f"  - {OUTPUT_PARQUET}")
    print(f"Training rows: {len(training_df)}")
    if "label_team1_win" in training_df.columns:
        print("Label distribution:")
        print(training_df["label_team1_win"].value_counts(dropna=False))


if __name__ == "__main__":
    main()