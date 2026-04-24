from __future__ import annotations

from pathlib import Path
import pandas as pd


TEAM_TABLE_PATH = Path("data/processed/training_table_team_v1.parquet")
VENUE_TABLE_PATH = Path("data/processed/training_table_venue_v1.parquet")

OUTPUT_DIR = Path("data/processed")
OUTPUT_CSV = OUTPUT_DIR / "training_table_v3.csv"
OUTPUT_PARQUET = OUTPUT_DIR / "training_table_v3.parquet"


BASE_METADATA_COLS = {
    "match_id",
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


def load_table(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")
    return pd.read_parquet(path)


def trim_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = ["match_id"] + [c for c in df.columns if c not in BASE_METADATA_COLS and c != "match_id"]
    return df[feature_cols].copy()


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    team_df = load_table(TEAM_TABLE_PATH, "team table")
    venue_df = load_table(VENUE_TABLE_PATH, "venue table")

    print(f"Loaded team table:  {team_df.shape}")
    print(f"Loaded venue table: {venue_df.shape}")

    venue_features = trim_feature_table(venue_df)

    merged = team_df.merge(
        venue_features,
        on="match_id",
        how="inner",
        validate="one_to_one",
    )

    if "label_team1_win" in merged.columns:
        merged = merged[merged["label_team1_win"].notna()].copy()
        merged["label_team1_win"] = merged["label_team1_win"].astype(int)

    if "is_binary_outcome" in merged.columns:
        merged = merged[merged["is_binary_outcome"] == True].copy()

    merged = merged.sort_values(
        ["date_start", "source_bucket", "competition_folder", "match_id"],
        na_position="last",
    ).reset_index(drop=True)

    merged.to_csv(OUTPUT_CSV, index=False)
    merged.to_parquet(OUTPUT_PARQUET, index=False)

    print("Done. Wrote:")
    print(f"  - {OUTPUT_CSV}")
    print(f"  - {OUTPUT_PARQUET}")
    print(f"Final shape: {merged.shape}")

    if "label_team1_win" in merged.columns:
        print("Label distribution:")
        print(merged["label_team1_win"].value_counts(dropna=False))


if __name__ == "__main__":
    main()