from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


MATCHES_MASTER_PATH = Path("data/interim/matches_master.parquet")
OUTPUT_DIR = Path("data/interim")
OUTPUT_CSV = OUTPUT_DIR / "player_match_stats.csv"
OUTPUT_PARQUET = OUTPUT_DIR / "player_match_stats.parquet"


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


def initialize_player_row(
    match_id: str,
    date_start: Any,
    player_name: str,
    team: str | None,
    opponent: str | None,
) -> dict[str, Any]:
    return {
        "match_id": match_id,
        "date_start": date_start,
        "player_name": player_name,
        "team": team,
        "opponent": opponent,
        # batting
        "innings_batted": 0,
        "runs_scored": 0,
        "balls_faced": 0,
        "fours": 0,
        "sixes": 0,
        "dismissed": 0,
        "dismissed_by_bowler": None,
        "dismissal_kind": None,
        "dismissal_fielders": None,
        # bowling
        "innings_bowled": 0,
        "balls_bowled": 0,
        "runs_conceded": 0,
        "wickets": 0,
        # fielding
        "catches": 0,
        "stumpings": 0,
        "run_out_direct": 0,
        "run_out_indirect": 0,
    }


def get_match_players(info: dict[str, Any]) -> dict[str, list[str]]:
    players = info.get("players", {})
    if isinstance(players, dict):
        return {team: roster for team, roster in players.items() if isinstance(roster, list)}
    return {}


def ensure_player(
    player_rows: dict[str, dict[str, Any]],
    match_id: str,
    date_start: Any,
    player_name: str | None,
    team: str | None,
    opponent: str | None,
) -> None:
    if not player_name:
        return
    if player_name not in player_rows:
        player_rows[player_name] = initialize_player_row(
            match_id=match_id,
            date_start=date_start,
            player_name=player_name,
            team=team,
            opponent=opponent,
        )


def parse_match_player_stats(match_row: pd.Series) -> list[dict[str, Any]]:
    file_path = Path(match_row["file_path"])
    match_id = match_row["match_id"]
    date_start = match_row["date_start"]

    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    info = data.get("info", {})
    teams = info.get("teams", []) if isinstance(info.get("teams", []), list) else []
    roster_map = get_match_players(info)
    team_opponent_map = get_team_opponent_map(teams)

    player_rows: dict[str, dict[str, Any]] = {}

    # Initialize from known team rosters
    for team, roster in roster_map.items():
        opponent = team_opponent_map.get(team)
        for player_name in roster:
            ensure_player(player_rows, match_id, date_start, player_name, team, opponent)

    innings_list = data.get("innings", [])
    if not isinstance(innings_list, list):
        return list(player_rows.values())

    for innings in innings_list:
        if not isinstance(innings, dict):
            continue

        batting_team = innings.get("team")
        bowling_team = team_opponent_map.get(batting_team)
        overs = innings.get("overs", [])
        if not isinstance(overs, list):
            continue

        batters_seen_this_innings: set[str] = set()
        bowlers_seen_this_innings: set[str] = set()

        for over_obj in overs:
            if not isinstance(over_obj, dict):
                continue

            deliveries = over_obj.get("deliveries", [])
            if not isinstance(deliveries, list):
                continue

            for ball in deliveries:
                if not isinstance(ball, dict):
                    continue

                batter = ball.get("batter")
                non_striker = ball.get("non_striker")
                bowler = ball.get("bowler")

                ensure_player(player_rows, match_id, date_start, batter, batting_team, bowling_team)
                ensure_player(player_rows, match_id, date_start, non_striker, batting_team, bowling_team)
                ensure_player(player_rows, match_id, date_start, bowler, bowling_team, batting_team)

                runs = ball.get("runs", {})
                batter_runs = safe_int(runs.get("batter"))
                extras_runs = safe_int(runs.get("extras"))

                # Batting
                if batter:
                    player_rows[batter]["runs_scored"] += batter_runs
                    player_rows[batter]["balls_faced"] += 1
                    if batter_runs == 4:
                        player_rows[batter]["fours"] += 1
                    elif batter_runs == 6:
                        player_rows[batter]["sixes"] += 1
                    batters_seen_this_innings.add(batter)

                # Bowling
                if bowler:
                    player_rows[bowler]["balls_bowled"] += 1
                    player_rows[bowler]["runs_conceded"] += batter_runs + extras_runs
                    bowlers_seen_this_innings.add(bowler)

                wickets = ball.get("wickets", [])
                if not isinstance(wickets, list):
                    continue

                for wicket in wickets:
                    if not isinstance(wicket, dict):
                        continue

                    dismissed_batter = wicket.get("player_out")
                    kind = wicket.get("kind")
                    fielders = wicket.get("fielders", [])

                    dismissal_fielder_names: list[str] = []
                    if isinstance(fielders, list):
                        for fielder in fielders:
                            if isinstance(fielder, dict):
                                fielder_name = fielder.get("name")
                            else:
                                fielder_name = fielder
                            if fielder_name:
                                dismissal_fielder_names.append(fielder_name)
                                ensure_player(
                                    player_rows,
                                    match_id,
                                    date_start,
                                    fielder_name,
                                    bowling_team,
                                    batting_team,
                                )

                    if dismissed_batter:
                        ensure_player(
                            player_rows,
                            match_id,
                            date_start,
                            dismissed_batter,
                            batting_team,
                            bowling_team,
                        )
                        player_rows[dismissed_batter]["dismissed"] += 1
                        player_rows[dismissed_batter]["dismissed_by_bowler"] = bowler
                        player_rows[dismissed_batter]["dismissal_kind"] = kind
                        player_rows[dismissed_batter]["dismissal_fielders"] = (
                            " | ".join(dismissal_fielder_names) if dismissal_fielder_names else None
                        )

                    credited_bowler_kinds = {
                        "bowled",
                        "caught",
                        "caught and bowled",
                        "lbw",
                        "stumped",
                        "hit wicket",
                    }
                    if bowler and kind in credited_bowler_kinds:
                        player_rows[bowler]["wickets"] += 1

                    for fielder_name in dismissal_fielder_names:
                        if kind in {"caught", "caught and bowled"}:
                            player_rows[fielder_name]["catches"] += 1
                        elif kind == "stumped":
                            player_rows[fielder_name]["stumpings"] += 1
                        elif kind == "run out":
                            if len(dismissal_fielder_names) == 1:
                                player_rows[fielder_name]["run_out_direct"] += 1
                            else:
                                player_rows[fielder_name]["run_out_indirect"] += 1

        for batter in batters_seen_this_innings:
            player_rows[batter]["innings_batted"] += 1

        for bowler in bowlers_seen_this_innings:
            player_rows[bowler]["innings_bowled"] += 1

    return list(player_rows.values())


def build_player_match_stats(matches_master_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for i, match_row in enumerate(matches_master_df.itertuples(index=False), start=1):
        match_series = pd.Series(match_row._asdict())
        match_rows = parse_match_player_stats(match_series)
        rows.extend(match_rows)

        if i % 250 == 0:
            print(f"Processed {i} matches... total player rows so far: {len(rows)}")

    df = pd.DataFrame(rows)

    if "date_start" in df.columns:
        df["date_start"] = pd.to_datetime(df["date_start"], errors="coerce")

    sort_cols = [c for c in ["date_start", "match_id", "team", "player_name"] if c in df.columns]
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
    player_match_stats_df = build_player_match_stats(matches_master_df)

    player_match_stats_df.to_csv(OUTPUT_CSV, index=False)
    player_match_stats_df.to_parquet(OUTPUT_PARQUET, index=False)

    print("Done. Wrote:")
    print(f"  - {OUTPUT_CSV}")
    print(f"  - {OUTPUT_PARQUET}")
    print(f"Player match stats rows: {len(player_match_stats_df)}")

    if "player_name" in player_match_stats_df.columns:
        print(f"Unique players: {player_match_stats_df['player_name'].nunique()}")


if __name__ == "__main__":
    main()