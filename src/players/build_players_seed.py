from __future__ import annotations

from pathlib import Path

import pandas as pd


PLAYER_MATCH_STATS_PATH = Path("data/interim/player_match_stats.parquet")
OLD_ENRICHED_CSV_PATH = Path("data/raw/external/t20s_players_wikipedia_enriched.csv")
OUTPUT_CSV_PATH = Path("data/interim/players_seed_v2.csv")


IMPORTANT_COLS = [
    "player_name",
    "cricsheet_id",
    "cricinfo_id",
    "playing_role",
    "batting_style",
    "bowling_style",
]


def normalize_name(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.lower()
    )


def normalize_id(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .replace({"nan": "", "None": "", "<NA>": ""})
    )


def build_universe() -> pd.DataFrame:
    df = pd.read_parquet(PLAYER_MATCH_STATS_PATH)

    keep_cols = ["player_name"]
    if "cricsheet_id" in df.columns:
        keep_cols.append("cricsheet_id")
    if "cricinfo_id" in df.columns:
        keep_cols.append("cricinfo_id")

    universe = df[keep_cols].dropna(subset=["player_name"]).drop_duplicates().copy()
    universe["player_name_norm"] = normalize_name(universe["player_name"])

    if "cricsheet_id" in universe.columns:
        universe["cricsheet_id_norm"] = normalize_id(universe["cricsheet_id"])
    else:
        universe["cricsheet_id"] = None
        universe["cricsheet_id_norm"] = ""

    if "cricinfo_id" in universe.columns:
        universe["cricinfo_id_norm"] = normalize_id(universe["cricinfo_id"])
    else:
        universe["cricinfo_id"] = None
        universe["cricinfo_id_norm"] = ""

    universe["has_cricsheet_id"] = (universe["cricsheet_id_norm"] != "").astype(int)
    universe = (
        universe.sort_values(["player_name_norm", "has_cricsheet_id"], ascending=[True, False])
        .drop_duplicates(subset=["player_name_norm"], keep="first")
        .drop(columns=["has_cricsheet_id"])
        .reset_index(drop=True)
    )
    return universe


def load_old_enriched() -> pd.DataFrame:
    old = pd.read_csv(OLD_ENRICHED_CSV_PATH).copy()

    if "player_name" not in old.columns:
        raise ValueError("Old enriched CSV must contain player_name")

    old["player_name_norm"] = normalize_name(old["player_name"])

    if "cricsheet_id" in old.columns:
        old["cricsheet_id_norm"] = normalize_id(old["cricsheet_id"])
    else:
        old["cricsheet_id"] = None
        old["cricsheet_id_norm"] = ""

    if "cricinfo_id" in old.columns:
        old["cricinfo_id_norm"] = normalize_id(old["cricinfo_id"])
    else:
        old["cricinfo_id"] = None
        old["cricinfo_id_norm"] = ""

    return old


def main() -> None:
    universe = build_universe()
    old = load_old_enriched()

    # Match by cricsheet_id first
    old_by_id = old.loc[old["cricsheet_id_norm"] != ""].copy()
    old_by_id = old_by_id.sort_values("player_name_norm").drop_duplicates("cricsheet_id_norm", keep="first")

    merged = universe.merge(
        old_by_id[
            [
                "cricsheet_id_norm",
                "cricinfo_id",
                "playing_role",
                "batting_style",
                "bowling_style",
            ]
        ],
        on="cricsheet_id_norm",
        how="left",
    )

    # Fallback match by name for rows still missing key fields
    old_by_name = old.sort_values("player_name_norm").drop_duplicates("player_name_norm", keep="first")
    merged = merged.merge(
        old_by_name[
            [
                "player_name_norm",
                "cricsheet_id",
                "cricinfo_id",
                "playing_role",
                "batting_style",
                "bowling_style",
            ]
        ],
        on="player_name_norm",
        how="left",
        suffixes=("", "_name"),
    )

    # Fill missing values from name fallback
    for col in ["cricsheet_id", "cricinfo_id", "playing_role", "batting_style", "bowling_style"]:
        fallback_col = f"{col}_name"
        if fallback_col in merged.columns:
            merged[col] = merged[col].combine_first(merged[fallback_col])

    # Keep only important fields
    out = merged[IMPORTANT_COLS].copy()

    # Deduplicate
    out["player_name_norm"] = normalize_name(out["player_name"])
    out = out.sort_values("player_name_norm").drop_duplicates("player_name_norm", keep="first")
    out = out.drop(columns=["player_name_norm"]).reset_index(drop=True)

    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV_PATH, index=False)

    print(f"Wrote: {OUTPUT_CSV_PATH}")
    print(f"Rows: {len(out)}")
    print(f"With cricsheet_id: {out['cricsheet_id'].notna().sum()}")
    print(f"With cricinfo_id: {out['cricinfo_id'].notna().sum()}")
    print(f"With playing_role: {out['playing_role'].notna().sum()}")
    print(f"With batting_style: {out['batting_style'].notna().sum()}")
    print(f"With bowling_style: {out['bowling_style'].notna().sum()}")


if __name__ == "__main__":
    main()