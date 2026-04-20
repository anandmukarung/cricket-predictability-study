from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


MATCHES_MASTER_PATH = Path("data/interim/matches_master.parquet")
VENUE_MATCH_STATS_PATH = Path("data/interim/venue_match_stats.parquet")
PLAYER_MATCH_STATS_V2_PATH = Path("data/interim/player_match_stats_v2.parquet")

OUTPUT_DIR = Path("data/processed")
OUTPUT_CSV = OUTPUT_DIR / "training_table_venue_v1.csv"
OUTPUT_PARQUET = OUTPUT_DIR / "training_table_venue_v1.parquet"


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def make_empty_venue_state() -> dict[str, Any]:
    return {
        "matches": 0,
        "first_innings_runs": 0,
        "second_innings_runs": 0,
        "chasing_wins": 0,
        "defending_wins": 0,
    }


def make_empty_team_venue_state() -> dict[str, Any]:
    return {
        "matches": 0,
        "wins": 0,
        "runs_scored": 0,
        "runs_conceded": 0,
    }


def make_empty_player_venue_state() -> dict[str, Any]:
    return {
        "matches": 0,
        "innings_batted": 0,
        "runs_scored": 0,
        "balls_faced": 0,
        "dismissed": 0,
        "innings_bowled": 0,
        "balls_bowled": 0,
        "runs_conceded": 0,
        "wickets": 0,
    }


def summarize_player_venue_state(state: dict[str, Any]) -> dict[str, float]:
    matches = state["matches"]
    runs_scored = state["runs_scored"]
    balls_faced = state["balls_faced"]
    dismissed = state["dismissed"]
    balls_bowled = state["balls_bowled"]
    runs_conceded = state["runs_conceded"]
    wickets = state["wickets"]

    return {
        "matches": float(matches),
        "batting_avg": safe_div(runs_scored, dismissed) if dismissed > 0 else float(runs_scored),
        "strike_rate": 100.0 * safe_div(runs_scored, balls_faced) if balls_faced > 0 else 0.0,
        "bowling_avg": safe_div(runs_conceded, wickets) if wickets > 0 else float(runs_conceded),
        "bowling_economy": 6.0 * safe_div(runs_conceded, balls_bowled) if balls_bowled > 0 else 0.0,
        "wickets_per_match": safe_div(wickets, matches),
    }


def aggregate_team_player_venue_features(
    match_player_rows: pd.DataFrame,
    team_name: str,
    venue: str,
    player_venue_state: dict[tuple[str, str, str], dict[str, Any]],
    prefix: str,
) -> dict[str, Any]:
    team_rows = match_player_rows.loc[match_player_rows["team"] == team_name].copy()
    team_rows = team_rows.sort_values(["player_name"]).drop_duplicates(subset=["player_name"], keep="first")

    player_summaries = []
    for _, row in team_rows.iterrows():
        key = (row["player_name"], row["cricsheet_id"], venue)
        summary = summarize_player_venue_state(player_venue_state[key])
        player_summaries.append(summary)

    if not player_summaries:
        return {
            f"{prefix}_xi_players_with_venue_history": 0,
            f"{prefix}_xi_avg_venue_experience": 0.0,
            f"{prefix}_xi_avg_venue_batting_avg": 0.0,
            f"{prefix}_xi_avg_venue_strike_rate": 0.0,
            f"{prefix}_xi_avg_venue_bowling_avg": 0.0,
            f"{prefix}_xi_avg_venue_bowling_economy": 0.0,
            f"{prefix}_xi_avg_venue_wickets_per_match": 0.0,
        }

    n = len(player_summaries)

    def avg(key: str) -> float:
        return safe_div(sum(p[key] for p in player_summaries), n)

    return {
        f"{prefix}_xi_players_with_venue_history": sum(1 for p in player_summaries if p["matches"] > 0),
        f"{prefix}_xi_avg_venue_experience": avg("matches"),
        f"{prefix}_xi_avg_venue_batting_avg": avg("batting_avg"),
        f"{prefix}_xi_avg_venue_strike_rate": avg("strike_rate"),
        f"{prefix}_xi_avg_venue_bowling_avg": avg("bowling_avg"),
        f"{prefix}_xi_avg_venue_bowling_economy": avg("bowling_economy"),
        f"{prefix}_xi_avg_venue_wickets_per_match": avg("wickets_per_match"),
    }


def build_training_table(
    matches_master_df: pd.DataFrame,
    venue_match_stats_df: pd.DataFrame,
    player_match_stats_df: pd.DataFrame,
) -> pd.DataFrame:
    matches_master_df = matches_master_df.sort_values(
        ["date_start", "source_bucket", "competition_folder", "file_name"],
        na_position="last",
    ).reset_index(drop=True)

    venue_match_stats_df = venue_match_stats_df.sort_values(
        ["date_start", "venue", "match_id"],
        na_position="last",
    ).reset_index(drop=True)

    player_match_stats_df = player_match_stats_df.sort_values(
        ["date_start", "match_id", "team", "player_name"],
        na_position="last",
    ).reset_index(drop=True)

    venue_rows_by_match = {
        match_id: grp.iloc[0]
        for match_id, grp in venue_match_stats_df.groupby("match_id", sort=False)
    }
    player_rows_by_match = {
        match_id: grp.reset_index(drop=True)
        for match_id, grp in player_match_stats_df.groupby("match_id", sort=False)
    }

    venue_state: dict[str, dict[str, Any]] = defaultdict(make_empty_venue_state)
    team_venue_state: dict[tuple[str, str], dict[str, Any]] = defaultdict(make_empty_team_venue_state)
    player_venue_state: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(make_empty_player_venue_state)

    training_rows: list[dict[str, Any]] = []

    for i, match_row in enumerate(matches_master_df.itertuples(index=False), start=1):
        match = match_row._asdict()
        match_id = match["match_id"]
        team1 = match.get("team1")
        team2 = match.get("team2")
        venue = match.get("venue")

        if not team1 or not team2 or not venue:
            continue

        venue_row = venue_rows_by_match.get(match_id)
        player_rows = player_rows_by_match.get(match_id)

        if venue_row is None or player_rows is None or player_rows.empty:
            continue

        vs = venue_state[venue]
        t1s = team_venue_state[(team1, venue)]
        t2s = team_venue_state[(team2, venue)]

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
            "venue": venue,
            "city": match.get("city"),
            "toss_winner": match.get("toss_winner"),
            "toss_decision": match.get("toss_decision"),
            "winner": match.get("winner"),
            "label_team1_win": int(match.get("winner") == team1) if pd.notna(match.get("winner")) else None,
            # venue-wide
            "venue_prior_matches": vs["matches"],
            "venue_prior_avg_first_innings_runs": safe_div(vs["first_innings_runs"], vs["matches"]),
            "venue_prior_avg_second_innings_runs": safe_div(vs["second_innings_runs"], vs["matches"]),
            "venue_prior_chasing_win_pct": safe_div(vs["chasing_wins"], vs["matches"]),
            "venue_prior_defending_win_pct": safe_div(vs["defending_wins"], vs["matches"]),
            # team at venue
            "team1_prior_matches_at_venue": t1s["matches"],
            "team1_prior_win_pct_at_venue": safe_div(t1s["wins"], t1s["matches"]),
            "team1_prior_avg_runs_scored_at_venue": safe_div(t1s["runs_scored"], t1s["matches"]),
            "team1_prior_avg_runs_conceded_at_venue": safe_div(t1s["runs_conceded"], t1s["matches"]),
            "team2_prior_matches_at_venue": t2s["matches"],
            "team2_prior_win_pct_at_venue": safe_div(t2s["wins"], t2s["matches"]),
            "team2_prior_avg_runs_scored_at_venue": safe_div(t2s["runs_scored"], t2s["matches"]),
            "team2_prior_avg_runs_conceded_at_venue": safe_div(t2s["runs_conceded"], t2s["matches"]),
        }

        row.update(
            aggregate_team_player_venue_features(
                player_rows, team1, venue, player_venue_state, "team1"
            )
        )
        row.update(
            aggregate_team_player_venue_features(
                player_rows, team2, venue, player_venue_state, "team2"
            )
        )

        diff_features = [
            "prior_matches_at_venue",
            "prior_win_pct_at_venue",
            "prior_avg_runs_scored_at_venue",
            "prior_avg_runs_conceded_at_venue",
            "xi_players_with_venue_history",
            "xi_avg_venue_experience",
            "xi_avg_venue_batting_avg",
            "xi_avg_venue_strike_rate",
            "xi_avg_venue_bowling_avg",
            "xi_avg_venue_bowling_economy",
            "xi_avg_venue_wickets_per_match",
        ]
        for feat in diff_features:
            row[f"diff_{feat}"] = row[f"team1_{feat}"] - row[f"team2_{feat}"]

        training_rows.append(row)

        # update venue-wide state
        vs["matches"] += 1
        vs["first_innings_runs"] += int(venue_row["first_innings_runs"])
        vs["second_innings_runs"] += int(venue_row["second_innings_runs"])
        vs["chasing_wins"] += int(bool(venue_row["chasing_team_won"]))
        vs["defending_wins"] += int(bool(venue_row["defending_team_won"]))

        # update team-at-venue state
        first_team = venue_row["first_innings_team"]
        second_team = venue_row["second_innings_team"]
        winner = venue_row["winner"]

        if pd.notna(first_team):
            s = team_venue_state[(first_team, venue)]
            s["matches"] += 1
            s["wins"] += int(winner == first_team)
            s["runs_scored"] += int(venue_row["first_innings_runs"])
            s["runs_conceded"] += int(venue_row["second_innings_runs"])

        if pd.notna(second_team):
            s = team_venue_state[(second_team, venue)]
            s["matches"] += 1
            s["wins"] += int(winner == second_team)
            s["runs_scored"] += int(venue_row["second_innings_runs"])
            s["runs_conceded"] += int(venue_row["first_innings_runs"])

        # update player-at-venue state
        for _, pr in player_rows.iterrows():
            key = (pr["player_name"], pr["cricsheet_id"], venue)
            ps = player_venue_state[key]
            ps["matches"] += 1
            ps["innings_batted"] += int(pr.get("innings_batted", 0))
            ps["runs_scored"] += int(pr.get("runs_scored", 0))
            ps["balls_faced"] += int(pr.get("balls_faced", 0))
            ps["dismissed"] += int(pr.get("dismissed", 0))
            ps["innings_bowled"] += int(pr.get("innings_bowled", 0))
            ps["balls_bowled"] += int(pr.get("balls_bowled", 0))
            ps["runs_conceded"] += int(pr.get("runs_conceded", 0))
            ps["wickets"] += int(pr.get("wickets", 0))

        if i % 250 == 0:
            print(f"Processed {i} matches into venue training rows...")

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
        raise FileNotFoundError(f"{MATCHES_MASTER_PATH} not found.")
    if not VENUE_MATCH_STATS_PATH.exists():
        raise FileNotFoundError(f"{VENUE_MATCH_STATS_PATH} not found.")
    if not PLAYER_MATCH_STATS_V2_PATH.exists():
        raise FileNotFoundError(f"{PLAYER_MATCH_STATS_V2_PATH} not found.")

    matches_master_df = pd.read_parquet(MATCHES_MASTER_PATH)
    venue_match_stats_df = pd.read_parquet(VENUE_MATCH_STATS_PATH)
    player_match_stats_df = pd.read_parquet(PLAYER_MATCH_STATS_V2_PATH)

    print(f"Loaded matches_master: {len(matches_master_df)} rows")
    print(f"Loaded venue_match_stats: {len(venue_match_stats_df)} rows")
    print(f"Loaded player_match_stats_v2: {len(player_match_stats_df)} rows")

    training_df = build_training_table(
        matches_master_df,
        venue_match_stats_df,
        player_match_stats_df,
    )

    training_df.to_csv(OUTPUT_CSV, index=False)
    training_df.to_parquet(OUTPUT_PARQUET, index=False)

    print("Done. Wrote:")
    print(f"  - {OUTPUT_CSV}")
    print(f"  - {OUTPUT_PARQUET}")
    print(f"Training rows: {len(training_df)}")


if __name__ == "__main__":
    main()