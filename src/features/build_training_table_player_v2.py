from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


MATCHES_MASTER_PATH = Path("data/interim/matches_master.parquet")
PLAYER_MATCH_STATS_V2_PATH = Path("data/interim/player_match_stats_v2.parquet")

OUTPUT_DIR = Path("data/processed")
OUTPUT_CSV = OUTPUT_DIR / "training_table_player_v2.csv"
OUTPUT_PARQUET = OUTPUT_DIR / "training_table_player_v2.parquet"


def safe_div(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return float(numerator) / float(denominator)


def make_empty_player_state() -> dict[str, Any]:
    return {
        "matches": 0,
        "innings_batted": 0,
        "batting_position_sum": 0.0,
        "batting_position_count": 0,
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
        # batting by phase
        "pp_balls_faced": 0,
        "pp_runs_scored": 0,
        "pp_dismissals": 0,
        "mid_balls_faced": 0,
        "mid_runs_scored": 0,
        "mid_dismissals": 0,
        "death_balls_faced": 0,
        "death_runs_scored": 0,
        "death_dismissals": 0,
        # bowling by phase
        "pp_balls_bowled": 0,
        "pp_runs_conceded": 0,
        "pp_wickets": 0,
        "mid_balls_bowled": 0,
        "mid_runs_conceded": 0,
        "mid_wickets": 0,
        "death_balls_bowled": 0,
        "death_runs_conceded": 0,
        "death_wickets": 0,
    }


def summarize_player_state(state: dict[str, Any]) -> dict[str, float]:
    matches = state["matches"]
    innings_batted = state["innings_batted"]
    batting_position_count = state["batting_position_count"]
    runs_scored = state["runs_scored"]
    balls_faced = state["balls_faced"]
    dismissed = state["dismissed"]
    innings_bowled = state["innings_bowled"]
    balls_bowled = state["balls_bowled"]
    runs_conceded = state["runs_conceded"]
    wickets = state["wickets"]

    summary = {
        "matches": float(matches),
        "innings_batted": float(innings_batted),
        "innings_bowled": float(innings_bowled),
        "runs_scored": float(runs_scored),
        "balls_faced": float(balls_faced),
        "dismissed": float(dismissed),
        "fours": float(state["fours"]),
        "sixes": float(state["sixes"]),
        "balls_bowled": float(balls_bowled),
        "runs_conceded": float(runs_conceded),
        "wickets": float(wickets),
        "catches": float(state["catches"]),
        "stumpings": float(state["stumpings"]),
        "run_out_direct": float(state["run_out_direct"]),
        "run_out_indirect": float(state["run_out_indirect"]),
        # overall batting
        "batting_avg": safe_div(runs_scored, dismissed) if dismissed > 0 else float(runs_scored),
        "strike_rate": 100.0 * safe_div(runs_scored, balls_faced) if balls_faced > 0 else 0.0,
        "balls_faced_per_match": safe_div(balls_faced, matches),
        "innings_batted_pct": safe_div(innings_batted, matches),
        "avg_batting_position": safe_div(state["batting_position_sum"], batting_position_count),
        # overall bowling
        "bowling_avg": safe_div(runs_conceded, wickets) if wickets > 0 else float(runs_conceded),
        "bowling_economy": 6.0 * safe_div(runs_conceded, balls_bowled) if balls_bowled > 0 else 0.0,
        "bowling_strike_rate": safe_div(balls_bowled, wickets) if wickets > 0 else float(balls_bowled),
        "balls_bowled_per_match": safe_div(balls_bowled, matches),
        "innings_bowled_pct": safe_div(innings_bowled, matches),
        "wickets_per_match": safe_div(wickets, matches),
    }

    # batting by phase
    for phase in ["pp", "mid", "death"]:
        phase_runs = state[f"{phase}_runs_scored"]
        phase_balls = state[f"{phase}_balls_faced"]
        phase_dismissals = state[f"{phase}_dismissals"]

        summary[f"{phase}_batting_avg"] = (
            safe_div(phase_runs, phase_dismissals) if phase_dismissals > 0 else float(phase_runs)
        )
        summary[f"{phase}_batting_strike_rate"] = (
            100.0 * safe_div(phase_runs, phase_balls) if phase_balls > 0 else 0.0
        )
        summary[f"{phase}_balls_faced_share"] = safe_div(phase_balls, balls_faced)

    # bowling by phase
    for phase in ["pp", "mid", "death"]:
        phase_runs = state[f"{phase}_runs_conceded"]
        phase_balls = state[f"{phase}_balls_bowled"]
        phase_wickets = state[f"{phase}_wickets"]

        summary[f"{phase}_bowling_avg"] = (
            safe_div(phase_runs, phase_wickets) if phase_wickets > 0 else float(phase_runs)
        )
        summary[f"{phase}_bowling_strike_rate"] = (
            safe_div(phase_balls, phase_wickets) if phase_wickets > 0 else float(phase_balls)
        )
        summary[f"{phase}_bowling_economy"] = (
            6.0 * safe_div(phase_runs, phase_balls) if phase_balls > 0 else 0.0
        )
        summary[f"{phase}_balls_bowled_share"] = safe_div(phase_balls, balls_bowled)

    return summary


def infer_role_flags(summary: dict[str, float]) -> dict[str, bool]:
    is_batter = (
        summary["innings_batted_pct"] >= 0.40
        or summary["balls_faced_per_match"] >= 6
        or (summary["avg_batting_position"] > 0 and summary["avg_batting_position"] <= 7.5)
    )

    is_bowler = (
        summary["innings_bowled_pct"] >= 0.30
        or summary["balls_bowled_per_match"] >= 6
        or summary["wickets_per_match"] >= 0.5
    )

    is_keeper = (
        summary["stumpings"] > 0
        or (summary["stumpings"] == 0 and summary["catches"] >= 20 and not is_bowler)
    )

    is_allrounder = is_batter and is_bowler

    return {
        "is_batter_inferred": bool(is_batter),
        "is_bowler_inferred": bool(is_bowler),
        "is_keeper_inferred": bool(is_keeper),
        "is_allrounder_inferred": bool(is_allrounder),
    }


def classify_batting_bucket(summary: dict[str, float]) -> str | None:
    pos = summary["avg_batting_position"]
    if pos <= 0:
        return None
    if pos <= 3:
        return "top"
    if pos <= 6:
        return "middle"
    return "lower"


def aggregate_group(player_summaries: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    if not player_summaries:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_avg_experience": 0.0,
            f"{prefix}_avg_batting_avg": 0.0,
            f"{prefix}_avg_strike_rate": 0.0,
            # batting phases
            f"{prefix}_pp_batting_avg": 0.0,
            f"{prefix}_pp_batting_strike_rate": 0.0,
            f"{prefix}_mid_batting_avg": 0.0,
            f"{prefix}_mid_batting_strike_rate": 0.0,
            f"{prefix}_death_batting_avg": 0.0,
            f"{prefix}_death_batting_strike_rate": 0.0,
            # bowling phases
            f"{prefix}_pp_bowling_avg": 0.0,
            f"{prefix}_pp_bowling_strike_rate": 0.0,
            f"{prefix}_pp_bowling_economy": 0.0,
            f"{prefix}_mid_bowling_avg": 0.0,
            f"{prefix}_mid_bowling_strike_rate": 0.0,
            f"{prefix}_mid_bowling_economy": 0.0,
            f"{prefix}_death_bowling_avg": 0.0,
            f"{prefix}_death_bowling_strike_rate": 0.0,
            f"{prefix}_death_bowling_economy": 0.0,
        }

    n = len(player_summaries)

    def avg(key: str) -> float:
        return safe_div(sum(p[key] for p in player_summaries), n)

    return {
        f"{prefix}_count": n,
        f"{prefix}_avg_experience": avg("matches"),
        f"{prefix}_avg_batting_avg": avg("batting_avg"),
        f"{prefix}_avg_strike_rate": avg("strike_rate"),
        f"{prefix}_pp_batting_avg": avg("pp_batting_avg"),
        f"{prefix}_pp_batting_strike_rate": avg("pp_batting_strike_rate"),
        f"{prefix}_mid_batting_avg": avg("mid_batting_avg"),
        f"{prefix}_mid_batting_strike_rate": avg("mid_batting_strike_rate"),
        f"{prefix}_death_batting_avg": avg("death_batting_avg"),
        f"{prefix}_death_batting_strike_rate": avg("death_batting_strike_rate"),
        f"{prefix}_pp_bowling_avg": avg("pp_bowling_avg"),
        f"{prefix}_pp_bowling_strike_rate": avg("pp_bowling_strike_rate"),
        f"{prefix}_pp_bowling_economy": avg("pp_bowling_economy"),
        f"{prefix}_mid_bowling_avg": avg("mid_bowling_avg"),
        f"{prefix}_mid_bowling_strike_rate": avg("mid_bowling_strike_rate"),
        f"{prefix}_mid_bowling_economy": avg("mid_bowling_economy"),
        f"{prefix}_death_bowling_avg": avg("death_bowling_avg"),
        f"{prefix}_death_bowling_strike_rate": avg("death_bowling_strike_rate"),
        f"{prefix}_death_bowling_economy": avg("death_bowling_economy"),
    }


def aggregate_team_features(
    match_player_rows: pd.DataFrame,
    player_state: dict[tuple[str, str], dict[str, Any]],
    team_name: str,
    prefix: str,
) -> dict[str, Any]:
    team_rows = match_player_rows.loc[match_player_rows["team"] == team_name].copy()
    team_rows = team_rows.sort_values(["player_name"]).drop_duplicates(subset=["player_name"], keep="first")

    player_summaries = []
    for _, row in team_rows.iterrows():
        key = (row["player_name"], row["cricsheet_id"])
        summary = summarize_player_state(player_state[key])
        summary.update(infer_role_flags(summary))
        summary["batting_bucket"] = classify_batting_bucket(summary)
        player_summaries.append(summary)

    batters = [p for p in player_summaries if p["is_batter_inferred"]]
    bowlers = [p for p in player_summaries if p["is_bowler_inferred"]]
    allrounders = [p for p in player_summaries if p["is_allrounder_inferred"]]
    keepers = [p for p in player_summaries if p["is_keeper_inferred"]]

    top_order = [p for p in batters if p["batting_bucket"] == "top"]
    middle_order = [p for p in batters if p["batting_bucket"] == "middle"]
    lower_order = [p for p in batters if p["batting_bucket"] == "lower"]

    out = {
        f"{prefix}_xi_size": len(player_summaries),
        f"{prefix}_xi_players_with_history": sum(1 for p in player_summaries if p["matches"] > 0),
        f"{prefix}_allrounder_count": len(allrounders),
        f"{prefix}_keeper_count": len(keepers),
    }

    out.update(aggregate_group(top_order, f"{prefix}_top_order"))
    out.update(aggregate_group(middle_order, f"{prefix}_middle_order"))
    out.update(aggregate_group(lower_order, f"{prefix}_lower_order"))
    out.update(aggregate_group(bowlers, f"{prefix}_bowlers"))

    return out


def update_player_state(state: dict[str, Any], player_row: pd.Series) -> None:
    state["matches"] += 1
    state["innings_batted"] += int(player_row.get("innings_batted", 0))
    state["runs_scored"] += int(player_row.get("runs_scored", 0))
    state["balls_faced"] += int(player_row.get("balls_faced", 0))
    state["dismissed"] += int(player_row.get("dismissed", 0))
    state["fours"] += int(player_row.get("fours", 0))
    state["sixes"] += int(player_row.get("sixes", 0))
    state["innings_bowled"] += int(player_row.get("innings_bowled", 0))
    state["balls_bowled"] += int(player_row.get("balls_bowled", 0))
    state["runs_conceded"] += int(player_row.get("runs_conceded", 0))
    state["wickets"] += int(player_row.get("wickets", 0))
    state["catches"] += int(player_row.get("catches", 0))
    state["stumpings"] += int(player_row.get("stumpings", 0))
    state["run_out_direct"] += int(player_row.get("run_out_direct", 0))
    state["run_out_indirect"] += int(player_row.get("run_out_indirect", 0))

    batting_position = player_row.get("batting_position")
    if pd.notna(batting_position):
        state["batting_position_sum"] += float(batting_position)
        state["batting_position_count"] += 1

    # batting phases
    for col in [
        "pp_balls_faced", "pp_runs_scored", "pp_dismissals",
        "mid_balls_faced", "mid_runs_scored", "mid_dismissals",
        "death_balls_faced", "death_runs_scored", "death_dismissals",
    ]:
        state[col] += int(player_row.get(col, 0))

    # bowling phases
    for col in [
        "pp_balls_bowled", "pp_runs_conceded", "pp_wickets",
        "mid_balls_bowled", "mid_runs_conceded", "mid_wickets",
        "death_balls_bowled", "death_runs_conceded", "death_wickets",
    ]:
        state[col] += int(player_row.get(col, 0))


def build_training_table(matches_master_df: pd.DataFrame, player_match_stats_df: pd.DataFrame) -> pd.DataFrame:
    matches_master_df = matches_master_df.sort_values(
        ["date_start", "source_bucket", "competition_folder", "file_name"],
        na_position="last",
    ).reset_index(drop=True)

    player_match_stats_df = player_match_stats_df.sort_values(
        ["date_start", "match_id", "team", "batting_position", "player_name"],
        na_position="last",
    ).reset_index(drop=True)

    player_rows_by_match = {
        match_id: grp.reset_index(drop=True)
        for match_id, grp in player_match_stats_df.groupby("match_id", sort=False)
    }

    player_state: dict[tuple[str, str], dict[str, Any]] = defaultdict(make_empty_player_state)
    training_rows: list[dict[str, Any]] = []

    for i, match_row in enumerate(matches_master_df.itertuples(index=False), start=1):
        match = match_row._asdict()
        match_id = match["match_id"]
        team1 = match.get("team1")
        team2 = match.get("team2")

        if not team1 or not team2:
            continue

        match_player_rows = player_rows_by_match.get(match_id)
        if match_player_rows is None or match_player_rows.empty:
            continue

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

        row.update(aggregate_team_features(match_player_rows, player_state, team1, "team1"))
        row.update(aggregate_team_features(match_player_rows, player_state, team2, "team2"))

        diff_features = [
            "xi_players_with_history",
            "allrounder_count",
            "keeper_count",
            "top_order_avg_batting_avg",
            "top_order_avg_strike_rate",
            "middle_order_avg_batting_avg",
            "middle_order_avg_strike_rate",
            "lower_order_avg_batting_avg",
            "lower_order_avg_strike_rate",
            "top_order_pp_batting_avg",
            "top_order_pp_batting_strike_rate",
            "middle_order_mid_batting_avg",
            "middle_order_mid_batting_strike_rate",
            "lower_order_death_batting_avg",
            "lower_order_death_batting_strike_rate",
            "bowlers_pp_bowling_avg",
            "bowlers_pp_bowling_strike_rate",
            "bowlers_pp_bowling_economy",
            "bowlers_mid_bowling_avg",
            "bowlers_mid_bowling_strike_rate",
            "bowlers_mid_bowling_economy",
            "bowlers_death_bowling_avg",
            "bowlers_death_bowling_strike_rate",
            "bowlers_death_bowling_economy",
        ]

        for feat in diff_features:
            row[f"diff_{feat}"] = row.get(f"team1_{feat}", 0.0) - row.get(f"team2_{feat}", 0.0)

        training_rows.append(row)

        # update only after writing row -> leak-proof
        for _, player_row in match_player_rows.iterrows():
            key = (player_row["player_name"], player_row["cricsheet_id"])
            update_player_state(player_state[key], player_row)

        if i % 250 == 0:
            print(f"Processed {i} matches into player v2 training rows...")

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
    if not PLAYER_MATCH_STATS_V2_PATH.exists():
        raise FileNotFoundError(f"{PLAYER_MATCH_STATS_V2_PATH} not found.")

    matches_master_df = pd.read_parquet(MATCHES_MASTER_PATH)
    player_match_stats_df = pd.read_parquet(PLAYER_MATCH_STATS_V2_PATH)

    print(f"Loaded matches_master: {len(matches_master_df)} rows")
    print(f"Loaded player_match_stats_v2: {len(player_match_stats_df)} rows")

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