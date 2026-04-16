from __future__ import annotations

from pathlib import Path

import pandas as pd


TEAM_TABLE_PATH = Path("data/processed/training_table_team_v1.parquet")
PLAYER_TABLE_PATH = Path("data/processed/training_table_player_v1.parquet")

OUTPUT_DIR = Path("data/processed")
OUTPUT_CSV = OUTPUT_DIR / "training_table_v1.csv"
OUTPUT_PARQUET = OUTPUT_DIR / "training_table_v1.parquet"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not TEAM_TABLE_PATH.exists():
        raise FileNotFoundError(f"{TEAM_TABLE_PATH} not found.")
    if not PLAYER_TABLE_PATH.exists():
        raise FileNotFoundError(f"{PLAYER_TABLE_PATH} not found.")

    team_df = pd.read_parquet(TEAM_TABLE_PATH)
    player_df = pd.read_parquet(PLAYER_TABLE_PATH)

    print(f"Loaded team table: {team_df.shape}")
    print(f"Loaded player table: {player_df.shape}")

    # Keep all team columns. From player table, keep only match_id plus
    # player-derived features to avoid duplicate metadata columns.
    player_feature_cols = [
        c for c in player_df.columns
        if c not in {
            "date_start",
            "source_bucket",
            "official_status",
            "competition_folder",
            "event_name",
            "match_type",
            "gender",
            "team1",
            "team2",
            "venue",
            "city",
            "toss_winner",
            "toss_decision",
            "winner",
            "label_team1_win",
        }
    ]
    player_df = player_df[player_feature_cols]

    merged_df = team_df.merge(
        player_df,
        on="match_id",
        how="inner",
        validate="one_to_one",
    )

    # Keep only clean labeled rows for modeling
    merged_df = merged_df[merged_df["label_team1_win"].notna()].copy()
    merged_df["label_team1_win"] = merged_df["label_team1_win"].astype(int)

    # Optional: keep only binary-outcome matches if present
    if "is_binary_outcome" in merged_df.columns:
        merged_df = merged_df[merged_df["is_binary_outcome"] == True].copy()

    merged_df = merged_df.sort_values(
        ["date_start", "source_bucket", "competition_folder", "match_id"],
        na_position="last",
    ).reset_index(drop=True)

    merged_df.to_csv(OUTPUT_CSV, index=False)
    merged_df.to_parquet(OUTPUT_PARQUET, index=False)

    print("Done. Wrote:")
    print(f"  - {OUTPUT_CSV}")
    print(f"  - {OUTPUT_PARQUET}")
    print(f"Final shape: {merged_df.shape}")
    print("Label distribution:")
    print(merged_df["label_team1_win"].value_counts(dropna=False))


if __name__ == "__main__":
    main()