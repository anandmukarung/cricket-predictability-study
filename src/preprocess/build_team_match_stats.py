from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


MATCHES_MASTER_PATH = Path("data/interim/matches_master.parquet")
OUTPUT_DIR = Path("data/interim")
OUTPUT_CSV = OUTPUT_DIR / "team_match_stats.csv"
OUTPUT_PARQUET = OUTPUT_DIR / "team_match_stats.parquet"


def safe_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def get_team_opponent_map(teams: list[str]) -> dict[str, str | None]:
    if len(teams) >= 2:
        return {teams[0]: teams[1], teams[1]: teams[0]}
    if len(teams) == 1:
        return {teams[0]: None}
    return {}


def summarize_innings(innings: dict[str, Any]) -> dict[str, Any]:
    batting_team = innings.get("team")
    overs = innings.get("overs", [])

    runs_scored = 0
    wickets_lost = 0
    legal_balls = 0

    if not isinstance(overs, list):
        return {
            "batting_team": batting_team,
            "runs_scored": 0,
            "wickets_lost": 0,
            "legal_balls": 0,
        }

    for over_obj in overs:
        if not isinstance(over_obj, dict):
            continue

        deliveries = over_obj.get("deliveries", [])
        if not isinstance(deliveries, list):
            continue

        for ball in deliveries:
            if not isinstance(ball, dict):
                continue

            runs = ball.get("runs", {})
            runs_scored += safe_int(runs.get("total"))

            extras = ball.get("extras", {})
            is_wide = "wides" in extras if isinstance(extras, dict) else False
            is_no_ball = "noballs" in extras if isinstance(extras, dict) else False

            if not is_wide and not is_no_ball:
                legal_balls += 1

            wickets = ball.get("wickets", [])
            if isinstance(wickets, list):
                wickets_lost += len(wickets)

    return {
        "batting_team": batting_team,
        "runs_scored": runs_scored,
        "wickets_lost": wickets_lost,
        "legal_balls": legal_balls,
    }


def parse_match_team_stats(match_row: pd.Series) -> list[dict[str, Any]]:
    file_path = Path(match_row["file_path"])

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    info = data.get("info", {})
    teams = info.get("teams", []) if isinstance(info.get("teams", []), list) else []
    team_opponent_map = get_team_opponent_map(teams)

    innings_list = data.get("innings", [])
    innings_summaries: list[dict[str, Any]] = []

    if isinstance(innings_list, list):
        for innings in innings_list:
            if isinstance(innings, dict):
                innings_summaries.append(summarize_innings(innings))

    # Aggregate by batting team in case of weird/multi-innings structures
    team_totals: dict[str, dict[str, int]] = {}
    for summary in innings_summaries:
        team = summary["batting_team"]
        if not team:
            continue
        if team not in team_totals:
            team_totals[team] = {
                "runs_scored": 0,
                "wickets_lost": 0,
                "legal_balls_faced": 0,
            }
        team_totals[team]["runs_scored"] += summary["runs_scored"]
        team_totals[team]["wickets_lost"] += summary["wickets_lost"]
        team_totals[team]["legal_balls_faced"] += summary["legal_balls"]

    rows: list[dict[str, Any]] = []

    for team in teams:
        opponent = team_opponent_map.get(team)
        team_stats = team_totals.get(
            team,
            {"runs_scored": 0, "wickets_lost": 0, "legal_balls_faced": 0},
        )
        opp_stats = team_totals.get(
            opponent,
            {"runs_scored": 0, "wickets_lost": 0, "legal_balls_faced": 0},
        )

        won = match_row["winner"] == team if pd.notna(match_row["winner"]) else False

        rows.append(
            {
                "match_id": match_row["match_id"],
                "date_start": match_row["date_start"],
                "team": team,
                "opponent": opponent,
                "source_bucket": match_row.get("source_bucket"),
                "official_status": match_row.get("official_status"),
                "competition_folder": match_row.get("competition_folder"),
                "event_name": match_row.get("event_name"),
                "match_type": match_row.get("match_type"),
                "gender": match_row.get("gender"),
                "venue": match_row.get("venue"),
                "city": match_row.get("city"),
                "toss_winner": match_row.get("toss_winner"),
                "toss_decision": match_row.get("toss_decision"),
                "winner": match_row.get("winner"),
                "won": won,
                "runs_scored": team_stats["runs_scored"],
                "runs_conceded": opp_stats["runs_scored"],
                "wickets_lost": team_stats["wickets_lost"],
                "wickets_taken": opp_stats["wickets_lost"],
                "legal_balls_faced": team_stats["legal_balls_faced"],
                "legal_balls_bowled": opp_stats["legal_balls_faced"],
            }
        )

    return rows


def build_team_match_stats(matches_master_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for i, match_row in enumerate(matches_master_df.itertuples(index=False), start=1):
        match_series = pd.Series(match_row._asdict())
        rows.extend(parse_match_team_stats(match_series))

        if i % 250 == 0:
            print(f"Processed {i} matches... total team rows so far: {len(rows)}")

    df = pd.DataFrame(rows)

    if "date_start" in df.columns:
        df["date_start"] = pd.to_datetime(df["date_start"], errors="coerce")

    sort_cols = [c for c in ["date_start", "match_id", "team"] if c in df.columns]
    df = df.sort_values(sort_cols, na_position="last").reset_index(drop=True)

    return df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MATCHES_MASTER_PATH.exists():
        raise FileNotFoundError(
            f"{MATCHES_MASTER_PATH} not found. Run build_matches_master.py first."
        )

    matches_master_df = pd.read_parquet(MATCHES_MASTER_PATH)

    print(f"Loaded matches_master with {len(matches_master_df)} matches")
    team_match_stats_df = build_team_match_stats(matches_master_df)

    team_match_stats_df.to_csv(OUTPUT_CSV, index=False)
    team_match_stats_df.to_parquet(OUTPUT_PARQUET, index=False)

    print("Done. Wrote:")
    print(f"  - {OUTPUT_CSV}")
    print(f"  - {OUTPUT_PARQUET}")
    print(f"Team match stats rows: {len(team_match_stats_df)}")
    print(f"Unique teams: {team_match_stats_df['team'].nunique()}")


if __name__ == "__main__":
    main()