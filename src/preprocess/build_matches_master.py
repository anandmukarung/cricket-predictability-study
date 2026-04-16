from __future__ import annotations

from pathlib import Path

import pandas as pd


INPUT_PARQUET = Path("data/interim/cricsheet_file_index.parquet")
OUTPUT_DIR = Path("data/interim")
OUTPUT_CSV = OUTPUT_DIR / "matches_master.csv"
OUTPUT_PARQUET = OUTPUT_DIR / "matches_master.parquet"


def build_matches_master(index_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the canonical one-row-per-match table used to drive chronological
    preprocessing and feature generation.

    For now, this keeps all indexed matches and only performs light cleaning
    and standardization. Filtering decisions can be added later.
    """
    matches_df = index_df.copy()

    text_cols = [
        "source_bucket",
        "official_status",
        "competition_folder",
        "event_name",
        "match_type",
        "gender",
        "team1",
        "team2",
        "winner",
        "result",
        "method",
        "eliminator",
        "venue",
        "city",
        "toss_winner",
        "toss_decision",
        "player_of_match",
        "file_name",
        "file_stem",
        "file_path",
        "match_id",
    ]
    for col in text_cols:
        if col in matches_df.columns:
            matches_df[col] = matches_df[col].where(matches_df[col].notna(), None)

    matches_df["has_winner"] = matches_df["winner"].notna() if "winner" in matches_df.columns else False
    matches_df["team1_win"] = (
        matches_df["winner"] == matches_df["team1"]
        if {"winner", "team1"}.issubset(matches_df.columns)
        else False
    )
    matches_df["team2_win"] = (
        matches_df["winner"] == matches_df["team2"]
        if {"winner", "team2"}.issubset(matches_df.columns)
        else False
    )
    matches_df["is_binary_outcome"] = (
        matches_df["winner"].isin(matches_df["team1"]) | matches_df["winner"].isin(matches_df["team2"])
        if {"winner", "team1", "team2"}.issubset(matches_df.columns)
        else False
    )

    preferred_order = [
        "match_id",
        "file_path",
        "file_name",
        "file_stem",
        "date_start",
        "date_end",
        "date_count",
        "source_bucket",
        "official_status",
        "competition_folder",
        "event_name",
        "match_type",
        "gender",
        "team1",
        "team2",
        "winner",
        "team1_win",
        "team2_win",
        "has_winner",
        "is_binary_outcome",
        "result",
        "by_runs",
        "by_wickets",
        "method",
        "eliminator",
        "venue",
        "city",
        "toss_winner",
        "toss_decision",
        "balls_per_over",
        "player_of_match",
        "teams_raw",
        "parse_error",
    ]
    ordered_cols = [c for c in preferred_order if c in matches_df.columns]
    remaining_cols = [c for c in matches_df.columns if c not in ordered_cols]
    matches_df = matches_df[ordered_cols + remaining_cols]

    sort_cols = [c for c in ["date_start", "source_bucket", "competition_folder", "file_name"] if c in matches_df.columns]
    matches_df = matches_df.sort_values(sort_cols, na_position="last").reset_index(drop=True)

    return matches_df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_PARQUET.exists():
        raise FileNotFoundError(
            f"{INPUT_PARQUET} not found. Run src/preprocess/indexer.py first."
        )

    index_df = pd.read_parquet(INPUT_PARQUET)
    matches_master_df = build_matches_master(index_df)

    matches_master_df.to_csv(OUTPUT_CSV, index=False)
    matches_master_df.to_parquet(OUTPUT_PARQUET, index=False)

    print("Done. Wrote:")
    print(f"  - {OUTPUT_CSV}")
    print(f"  - {OUTPUT_PARQUET}")
    print(f"Matches master rows: {len(matches_master_df)}")


if __name__ == "__main__":
    main()