from __future__ import annotations

from pathlib import Path

import pandas as pd


PLAYER_MATCH_STATS_PATH = Path("data/interim/player_match_stats.parquet")
ENRICHED_PLAYERS_CSV_PATH = Path("data/raw/external/t20s_players_wikipedia_enriched.csv")

OUTPUT_DIR = Path("data/interim")
MISSING_CSV = OUTPUT_DIR / "missing_players_v2.csv"
UNRESOLVED_CSV = OUTPUT_DIR / "unresolved_players_v2.csv"
INCOMPLETE_CSV = OUTPUT_DIR / "incomplete_players_v2.csv"
PLAYERS_TO_ENRICH_CSV = OUTPUT_DIR / "players_to_enrich_v2.csv"


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
        .replace({"nan": "", "None": ""})
    )


def load_player_universe() -> pd.DataFrame:
    if not PLAYER_MATCH_STATS_PATH.exists():
        raise FileNotFoundError(f"{PLAYER_MATCH_STATS_PATH} not found")

    df = pd.read_parquet(PLAYER_MATCH_STATS_PATH)

    required_cols = {"player_name"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"player_match_stats missing required columns: {missing}")

    keep_cols = ["player_name"]
    if "cricsheet_id" in df.columns:
        keep_cols.append("cricsheet_id")

    universe = df[keep_cols].drop_duplicates().copy()
    universe["player_name_norm"] = normalize_name(universe["player_name"])

    if "cricsheet_id" in universe.columns:
        universe["cricsheet_id_norm"] = normalize_id(universe["cricsheet_id"])
    else:
        universe["cricsheet_id_norm"] = ""

    # Prefer rows that actually have cricsheet_id when deduplicating by player name
    universe["has_id"] = (universe["cricsheet_id_norm"] != "").astype(int)
    universe = (
        universe.sort_values(["player_name_norm", "has_id"], ascending=[True, False])
        .drop_duplicates(subset=["player_name_norm"], keep="first")
        .drop(columns=["has_id"])
        .reset_index(drop=True)
    )

    return universe


def load_enriched_players() -> pd.DataFrame:
    if not ENRICHED_PLAYERS_CSV_PATH.exists():
        raise FileNotFoundError(f"{ENRICHED_PLAYERS_CSV_PATH} not found")

    enriched = pd.read_csv(ENRICHED_PLAYERS_CSV_PATH)

    if "player_name" not in enriched.columns:
        raise ValueError("enriched CSV must contain 'player_name'")

    enriched = enriched.copy()
    enriched["player_name_norm"] = normalize_name(enriched["player_name"])

    if "cricsheet_id" in enriched.columns:
        enriched["cricsheet_id_norm"] = normalize_id(enriched["cricsheet_id"])
    else:
        enriched["cricsheet_id_norm"] = ""

    return enriched


def infer_unresolved_mask(enriched_df: pd.DataFrame) -> pd.Series:
    if "resolved" in enriched_df.columns:
        return ~enriched_df["resolved"].astype(str).str.lower().isin({"true", "1", "yes"})
    if "is_resolved" in enriched_df.columns:
        return ~enriched_df["is_resolved"].astype(str).str.lower().isin({"true", "1", "yes"})
    if "wiki_url" in enriched_df.columns:
        return enriched_df["wiki_url"].isna() | (enriched_df["wiki_url"].astype(str).str.strip() == "")
    return pd.Series(False, index=enriched_df.index)


def infer_incomplete_mask(enriched_df: pd.DataFrame) -> pd.Series:
    role_missing = (
        enriched_df["playing_role"].isna() | (enriched_df["playing_role"].astype(str).str.strip() == "")
        if "playing_role" in enriched_df.columns
        else pd.Series(True, index=enriched_df.index)
    )
    batting_missing = (
        enriched_df["batting_style"].isna() | (enriched_df["batting_style"].astype(str).str.strip() == "")
        if "batting_style" in enriched_df.columns
        else pd.Series(True, index=enriched_df.index)
    )
    bowling_missing = (
        enriched_df["bowling_style"].isna() | (enriched_df["bowling_style"].astype(str).str.strip() == "")
        if "bowling_style" in enriched_df.columns
        else pd.Series(True, index=enriched_df.index)
    )

    return role_missing | (batting_missing & bowling_missing)


def build_match_flags(universe_df: pd.DataFrame, enriched_df: pd.DataFrame) -> pd.DataFrame:
    """
    Matching priority:
    1. cricsheet_id_norm if universe row has non-empty id
    2. fallback to player_name_norm
    """
    enriched_ids = set(enriched_df.loc[enriched_df["cricsheet_id_norm"] != "", "cricsheet_id_norm"])
    enriched_names = set(enriched_df["player_name_norm"])

    universe_df = universe_df.copy()

    universe_df["matched_by_id"] = universe_df["cricsheet_id_norm"].isin(enriched_ids) & (universe_df["cricsheet_id_norm"] != "")
    universe_df["matched_by_name"] = universe_df["player_name_norm"].isin(enriched_names)

    universe_df["is_matched"] = universe_df["matched_by_id"] | universe_df["matched_by_name"]

    return universe_df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    universe_df = load_player_universe()
    enriched_df = load_enriched_players()

    universe_df = build_match_flags(universe_df, enriched_df)

    # Missing entirely from enrichment
    missing_df = universe_df.loc[~universe_df["is_matched"]].copy()

    # Figure out which enriched rows correspond to current universe
    current_ids = set(universe_df.loc[universe_df["cricsheet_id_norm"] != "", "cricsheet_id_norm"])
    current_names = set(universe_df["player_name_norm"])

    enriched_current_mask = (
        ((enriched_df["cricsheet_id_norm"] != "") & enriched_df["cricsheet_id_norm"].isin(current_ids))
        | enriched_df["player_name_norm"].isin(current_names)
    )

    unresolved_df = enriched_df.loc[enriched_current_mask & infer_unresolved_mask(enriched_df)].copy()
    incomplete_df = enriched_df.loc[enriched_current_mask & infer_incomplete_mask(enriched_df)].copy()

    missing_combined = missing_df[["player_name", "player_name_norm", "cricsheet_id_norm"]].copy()
    missing_combined["reason"] = "missing_entirely"

    unresolved_combined = unresolved_df[["player_name", "player_name_norm", "cricsheet_id_norm"]].copy()
    unresolved_combined["reason"] = "unresolved"

    incomplete_combined = incomplete_df[["player_name", "player_name_norm", "cricsheet_id_norm"]].copy()
    incomplete_combined["reason"] = "incomplete_fields"

    combined = pd.concat(
        [missing_combined, unresolved_combined, incomplete_combined],
        ignore_index=True,
    ).drop_duplicates(subset=["cricsheet_id_norm", "player_name_norm", "reason"])

    reason_summary = (
        combined.groupby(["player_name", "player_name_norm", "cricsheet_id_norm"], dropna=False)["reason"]
        .apply(lambda s: " | ".join(sorted(set(s))))
        .reset_index()
        .sort_values(["player_name_norm"])
        .reset_index(drop=True)
    )

    missing_df.to_csv(MISSING_CSV, index=False)
    unresolved_df.to_csv(UNRESOLVED_CSV, index=False)
    incomplete_df.to_csv(INCOMPLETE_CSV, index=False)
    reason_summary.to_csv(PLAYERS_TO_ENRICH_CSV, index=False)

    print("Done. Wrote:")
    print(f"  - {MISSING_CSV}")
    print(f"  - {UNRESOLVED_CSV}")
    print(f"  - {INCOMPLETE_CSV}")
    print(f"  - {PLAYERS_TO_ENRICH_CSV}")
    print()
    print(f"Current player universe: {len(universe_df)}")
    print(f"Matched by cricsheet_id: {int(universe_df['matched_by_id'].sum())}")
    print(f"Matched by name fallback: {int((~universe_df['matched_by_id'] & universe_df['matched_by_name']).sum())}")
    print(f"Missing entirely: {len(missing_df)}")
    print(f"Unresolved: {len(unresolved_df)}")
    print(f"Incomplete for v2: {len(incomplete_df)}")
    print(f"Combined players needing enrichment: {len(reason_summary)}")


if __name__ == "__main__":
    main()