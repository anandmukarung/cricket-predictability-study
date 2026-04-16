from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


MATCHES_MASTER_PATH = Path("data/interim/matches_master.parquet")
PLAYER_MATCH_STATS_PATH = Path("data/interim/player_match_stats.parquet")

OUTPUT_DIR = Path("data/processed")
OUTPUT_CSV = OUTPUT_DIR / "training_table_player_v1.csv"
OUTPUT_PARQUET = OUTPUT_DIR / "training_table_player_v1.parquet"


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def make_empty_player_state() -> dict[str, Any]:
    return {
        "matches": 0,
        "innings_batted": 0,
        "runs_scored": 0,
        "balls_faced": 0,
        "dismissed": 0,
        "fours": 0,
        "sixes": 0,
        "innings_bowled": 0,
        "balls_bowled": 0,
        "runs_conceded": 0,
        "wickets": 0,
        "catches": 0,
        "stumpings": 0,
        "run_out_direct": 0,
        "run_out_indirect": 0,
    }


def extract_match_rosters(file_path: str) -> dict[str, list[str]]:
    path = Path(file_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    info = data.get("info", {})
    players = info.get("players", {})
    if isinstance(players, dict):
        return {
            team: roster
            for team, roster in players.items()
            if isinstance(roster, list)
        }
    return {}


def summarize_player_state(state: dict[str, Any]) -> dict[str, float]:
    matches = state["matches"]
    innings_batted = state["innings_batted"]
    runs_scored = state["runs_scored"]
    balls_faced = state["balls_faced"]
    dismissed = state["dismissed"]
    innings_bowled = state["innings_bowled"]
    balls_bowled = state["balls_bowled"]
    runs_conceded = state["runs_conceded"]
    wickets = state["wickets"]

    batting_avg = safe_div(runs_scored, dismissed) if dismissed > 0 else float(runs_scored)
    strike_rate = 100.0 * safe_div(runs_scored, balls_faced) if balls_faced > 0 else 0.0
    bowling_avg = safe_div(runs_conceded, wickets) if wickets > 0 else float(runs_conceded)
    bowling_economy = 6.0 * safe_div(runs_conceded, balls_bowled) if balls_bowled > 0 else 0.0
    wickets_per_match = safe_div(wickets, matches)

    return {
        "matches": float(matches),
        "innings_batted": float(innings_batted),
        "runs_scored": float(runs_scored),
        "balls_faced": float(balls_faced),
        "dismissed": float(dismissed),
        "fours": float(state["fours"]),
        "sixes": float(state["sixes"]),
        "innings_bowled": float(innings_bowled),
        "balls_bowled": float(balls_bowled),
        "runs_conceded": float(runs_conceded),
        "wickets": float(wickets),
        "batting_avg": batting_avg,
        "strike_rate": strike_rate,
        "bowling_avg": bowling_avg,
        "bowling_economy": bowling_economy,
        "wickets_per_match": wickets_per_match,
        "catches": float(state["catches"]),
        "stumpings": float(state["stumpings"]),
        "run_out_direct": float(state["run_out_direct"]),
        "run_out_indirect": float(state["run_out_indirect"]),
    }


def aggregate_team_player_features(
    roster: list[str],
    player_state: dict[str, dict[str, Any]],
    prefix: str,
) -> dict[str, Any]:
    if not roster:
        return {
            f"{prefix}_xi_size": 0,
            f"{prefix}_xi_players_with_history": 0,
            f"{prefix}_xi_avg_experience": 0.0,
            f"{prefix}_xi_max_experience": 0.0,
            f"{prefix}_xi_avg_batting_avg": 0.0,
            f"{prefix}_xi_avg_strike_rate": 0.0,
            f"{prefix}_xi_avg_bowling_avg": 0.0,
            f"{prefix}_xi_avg_bowling_economy": 0.0,
            f"{prefix}_xi_avg_wickets_per_match": 0.0,
            f"{prefix}_xi_avg_catches": 0.0,
        }

    player_summaries = []
    for player_name in roster:
        state = player_state[player_name]
        summary = summarize_player_state(state)
        summary["player_name"] = player_name
        player_summaries.append(summary)

    xi_size = len(player_summaries)
    players_with_history = sum(1 for p in player_summaries if p["matches"] > 0)

    def avg(key: str) -> float:
        return safe_div(sum(p[key] for p in player_summaries), xi_size)

    def max_val(key: str) -> float:
        return max((p[key] for p in player_summaries), default=0.0)

    return {
        f"{prefix}_xi_size": xi_size,
        f"{prefix}_xi_players_with_history": players_with_history,
        f"{prefix}_xi_avg_experience": avg("matches"),
        f"{prefix}_xi_max_experience": max_val("matches"),
        f"{prefix}_xi_avg_batting_avg": avg("batting_avg"),
        f"{prefix}_xi_avg_strike_rate": avg("strike_rate"),
        f"{prefix}_xi_avg_bowling_avg": avg("bowling_avg"),
        f"{prefix}_xi_avg_bowling_economy": avg("bowling_economy"),
        f"{prefix}_xi_avg_wickets_per_match": avg("wickets_per_match"),
        f"{prefix}_xi_avg_catches": avg("catches"),
    }


def update_player_state(state: dict[str, Any], player_row: pd.Series) -> None:
    for key in [
        "matches",
        "innings_batted",
        "runs_scored",
        "balls_faced",
        "dismissed",
        "fours",
        "sixes",
        "innings_bowled",
        "balls_bowled",
        "runs_conceded",
        "wickets",
        "catches",
        "stumpings",
        "run_out_direct",
        "run_out_indirect",
    ]:
        if key == "matches":
            state[key] += 1
        else:
            state[key] += int(player_row.get(key, 0))


def build_training_table(
    matches_master_df: pd.DataFrame,
    player_match_stats_df: pd.DataFrame,
) -> pd.DataFrame:
    matches_master_df = matches_master_df.sort_values(
        ["date_start", "source_bucket", "competition_folder", "file_name"],
        na_position="last",
    ).reset_index(drop=True)

    player_match_stats_df = player_match_stats_df.sort_values(
        ["date_start", "match_id", "team", "player_name"],
        na_position="last",
    ).reset_index(drop=True)

    player_rows_by_match: dict[str, pd.DataFrame] = {
        match_id: grp.reset_index(drop=True)
        for match_id, grp in player_match_stats_df.groupby("match_id", sort=False)
    }

    player_state: dict[str, dict[str, Any]] = defaultdict(make_empty_player_state)
    training_rows: list[dict[str, Any]] = []

    for i, match_row in enumerate(matches_master_df.itertuples(index=False), start=1):
        match = match_row._asdict()
        match_id = match["match_id"]
        team1 = match.get("team1")
        team2 = match.get("team2")

        if not team1 or not team2:
            continue

        roster_map = extract_match_rosters(match["file_path"])
        team1_roster = roster_map.get(team1, [])
        team2_roster = roster_map.get(team2, [])

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
        }

        row.update(aggregate_team_player_features(team1_roster, player_state, "team1"))
        row.update(aggregate_team_player_features(team2_roster, player_state, "team2"))

        # Difference features
        row["diff_xi_players_with_history"] = row["team1_xi_players_with_history"] - row["team2_xi_players_with_history"]
        row["diff_xi_avg_experience"] = row["team1_xi_avg_experience"] - row["team2_xi_avg_experience"]
        row["diff_xi_max_experience"] = row["team1_xi_max_experience"] - row["team2_xi_max_experience"]
        row["diff_xi_avg_batting_avg"] = row["team1_xi_avg_batting_avg"] - row["team2_xi_avg_batting_avg"]
        row["diff_xi_avg_strike_rate"] = row["team1_xi_avg_strike_rate"] - row["team2_xi_avg_strike_rate"]
        row["diff_xi_avg_bowling_avg"] = row["team1_xi_avg_bowling_avg"] - row["team2_xi_avg_bowling_avg"]
        row["diff_xi_avg_bowling_economy"] = row["team1_xi_avg_bowling_economy"] - row["team2_xi_avg_bowling_economy"]
        row["diff_xi_avg_wickets_per_match"] = row["team1_xi_avg_wickets_per_match"] - row["team2_xi_avg_wickets_per_match"]
        row["diff_xi_avg_catches"] = row["team1_xi_avg_catches"] - row["team2_xi_avg_catches"]

        training_rows.append(row)

        # Update state only after writing pre-match row
        match_player_rows = player_rows_by_match.get(match_id)
        if match_player_rows is not None and not match_player_rows.empty:
            for _, player_row in match_player_rows.iterrows():
                player_name = player_row["player_name"]
                update_player_state(player_state[player_name], player_row)

        if i % 250 == 0:
            print(f"Processed {i} matches into player training rows...")

    training_df = pd.DataFrame(training_rows)
    if "date_start" in training_df.columns:
        training_df["date_start"] = pd.to_datetime(training_df["date_start"], errors="coerce")

    training_df = training_df.sort_values(
        ["date_start", "source_bucket", "competition_folder", "match_id"],
        na_position="last",
    ).reset_index(drop=True)

    return training_df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MATCHES_MASTER_PATH.exists():
        raise FileNotFoundError(f"{MATCHES_MASTER_PATH} not found. Build matches_master first.")
    if not PLAYER_MATCH_STATS_PATH.exists():
        raise FileNotFoundError(f"{PLAYER_MATCH_STATS_PATH} not found. Build player_match_stats first.")

    matches_master_df = pd.read_parquet(MATCHES_MASTER_PATH)
    player_match_stats_df = pd.read_parquet(PLAYER_MATCH_STATS_PATH)

    print(f"Loaded matches_master: {len(matches_master_df)} rows")
    print(f"Loaded player_match_stats: {len(player_match_stats_df)} rows")

    training_df = build_training_table(matches_master_df, player_match_stats_df)

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