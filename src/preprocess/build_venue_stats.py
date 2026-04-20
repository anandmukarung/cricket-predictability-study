from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


MATCHES_MASTER_PATH = Path("data/interim/matches_master.parquet")
OUTPUT_DIR = Path("data/interim")
OUTPUT_CSV = OUTPUT_DIR / "venue_match_stats.csv"
OUTPUT_PARQUET = OUTPUT_DIR / "venue_match_stats.parquet"


def safe_int(value: Any) -> int:
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


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


def parse_match_venue_stats(match_row: pd.Series) -> dict[str, Any]:
    file_path = Path(match_row["file_path"])

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    innings_list = data.get("innings", [])
    innings_summaries = []

    if isinstance(innings_list, list):
        for innings in innings_list:
            if isinstance(innings, dict):
                innings_summaries.append(summarize_innings(innings))

    first = innings_summaries[0] if len(innings_summaries) > 0 else {}
    second = innings_summaries[1] if len(innings_summaries) > 1 else {}

    first_team = first.get("batting_team")
    second_team = second.get("batting_team")
    winner = match_row.get("winner")

    chasing_team_won = bool(second_team and winner == second_team)
    defending_team_won = bool(first_team and winner == first_team)

    return {
        "match_id": match_row["match_id"],
        "date_start": match_row["date_start"],
        "venue": match_row.get("venue"),
        "city": match_row.get("city"),
        "source_bucket": match_row.get("source_bucket"),
        "official_status": match_row.get("official_status"),
        "competition_folder": match_row.get("competition_folder"),
        "event_name": match_row.get("event_name"),
        "match_type": match_row.get("match_type"),
        "gender": match_row.get("gender"),
        "winner": winner,
        "first_innings_team": first_team,
        "first_innings_runs": first.get("runs_scored", 0),
        "first_innings_wickets_lost": first.get("wickets_lost", 0),
        "first_innings_legal_balls": first.get("legal_balls", 0),
        "second_innings_team": second_team,
        "second_innings_runs": second.get("runs_scored", 0),
        "second_innings_wickets_lost": second.get("wickets_lost", 0),
        "second_innings_legal_balls": second.get("legal_balls", 0),
        "chasing_team_won": chasing_team_won,
        "defending_team_won": defending_team_won,
    }


def build_venue_match_stats(matches_master_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for i, match_row in enumerate(matches_master_df.itertuples(index=False), start=1):
        match_series = pd.Series(match_row._asdict())
        rows.append(parse_match_venue_stats(match_series))

        if i % 250 == 0:
            print(f"Processed {i} matches into venue rows...")

    df = pd.DataFrame(rows)

    if "date_start" in df.columns:
        df["date_start"] = pd.to_datetime(df["date_start"], errors="coerce")

    df = df.sort_values(
        ["date_start", "venue", "match_id"],
        na_position="last",
    ).reset_index(drop=True)

    return df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not MATCHES_MASTER_PATH.exists():
        raise FileNotFoundError(f"{MATCHES_MASTER_PATH} not found.")

    matches_master_df = pd.read_parquet(MATCHES_MASTER_PATH)

    print(f"Loaded matches_master with {len(matches_master_df)} matches")
    venue_match_stats_df = build_venue_match_stats(matches_master_df)

    venue_match_stats_df.to_csv(OUTPUT_CSV, index=False)
    venue_match_stats_df.to_parquet(OUTPUT_PARQUET, index=False)

    print("Done. Wrote:")
    print(f"  - {OUTPUT_CSV}")
    print(f"  - {OUTPUT_PARQUET}")
    print(f"Venue rows: {len(venue_match_stats_df)}")
    print(f"Unique venues: {venue_match_stats_df['venue'].nunique(dropna=True)}")


if __name__ == "__main__":
    main()