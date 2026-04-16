from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


RAW_ROOT = Path("data/raw/Cricsheet")
OUTPUT_DIR = Path("data/interim")
OUTPUT_CSV = OUTPUT_DIR / "cricsheet_file_index.csv"
OUTPUT_PARQUET = OUTPUT_DIR / "cricsheet_file_index.parquet"


def infer_source_metadata(file_path: Path, raw_root: Path) -> dict[str, Any]:
    """
    Infer source info from the folder structure.

    Expected structure examples:
    - data/raw/Cricsheet/International/t20Internationals_official/<file>.json
    - data/raw/Cricsheet/International/t20Internationals_unofficial/<file>.json
    - data/raw/Cricsheet/Leagues/IPL/<file>.json
    """
    rel_parts = file_path.relative_to(raw_root).parts

    source_bucket = None
    official_status = None
    competition_folder = None

    if len(rel_parts) >= 2 and rel_parts[0].lower() == "international":
        source_bucket = "international"
        competition_folder = rel_parts[1]

        if "official" in rel_parts[1].lower():
            official_status = "official"
        elif "unofficial" in rel_parts[1].lower():
            official_status = "unofficial"
        else:
            official_status = "unknown"

    elif len(rel_parts) >= 2 and rel_parts[0].lower() == "leagues":
        source_bucket = "league"
        official_status = "league"
        competition_folder = rel_parts[1]

    else:
        source_bucket = "unknown"
        official_status = "unknown"
        competition_folder = rel_parts[0] if rel_parts else None

    return {
        "source_bucket": source_bucket,
        "official_status": official_status,
        "competition_folder": competition_folder,
    }


def extract_info_value(info: dict[str, Any], *keys: str, default: Any = None) -> Any:
    current: Any = info
    for key in keys:
        if not isinstance(current, dict):
            return default
        current = current.get(key)
        if current is None:
            return default
    return current


def safe_join(values: list[Any] | None) -> str | None:
    if not values:
        return None
    return " | ".join(str(v) for v in values)


def extract_match_metadata(file_path: Path) -> dict[str, Any]:
    with file_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    info = data.get("info", {})

    dates = info.get("dates", [])
    teams = info.get("teams", [])
    event_name = extract_info_value(info, "event", "name")
    match_type = info.get("match_type")
    gender = info.get("gender")
    venue = info.get("venue")
    city = info.get("city")
    balls_per_over = info.get("balls_per_over")
    player_of_match = info.get("player_of_match")
    outcome = info.get("outcome", {})
    toss = info.get("toss", {})

    winner = outcome.get("winner")
    result = outcome.get("result")
    by_runs = extract_info_value(outcome, "by", "runs")
    by_wickets = extract_info_value(outcome, "by", "wickets")
    method = outcome.get("method")
    eliminator = outcome.get("eliminator")

    toss_winner = toss.get("winner")
    toss_decision = toss.get("decision")

    team1 = teams[0] if len(teams) > 0 else None
    team2 = teams[1] if len(teams) > 1 else None

    return {
        "file_name": file_path.name,
        "file_stem": file_path.stem,
        "file_path": str(file_path),
        "match_id": file_path.stem,
        "date_start": dates[0] if dates else None,
        "date_end": dates[-1] if dates else None,
        "date_count": len(dates),
        "team1": team1,
        "team2": team2,
        "teams_raw": safe_join(teams),
        "event_name": event_name,
        "match_type": match_type,
        "gender": gender,
        "venue": venue,
        "city": city,
        "balls_per_over": balls_per_over,
        "player_of_match": safe_join(player_of_match),
        "winner": winner,
        "result": result,
        "by_runs": by_runs,
        "by_wickets": by_wickets,
        "method": method,
        "eliminator": eliminator,
        "toss_winner": toss_winner,
        "toss_decision": toss_decision,
    }


def build_index(raw_root: Path) -> pd.DataFrame:
    json_files = sorted(raw_root.rglob("*.json"))

    if not json_files:
        raise FileNotFoundError(f"No JSON files found under {raw_root}")

    rows: list[dict[str, Any]] = []

    for i, file_path in enumerate(json_files, start=1):
        try:
            source_meta = infer_source_metadata(file_path, raw_root)
            match_meta = extract_match_metadata(file_path)
            row = {**source_meta, **match_meta}
            rows.append(row)

            if i % 500 == 0:
                print(f"Processed {i} files...")

        except Exception as e:
            rows.append(
                {
                    "source_bucket": "error",
                    "official_status": "error",
                    "competition_folder": None,
                    "file_name": file_path.name,
                    "file_stem": file_path.stem,
                    "file_path": str(file_path),
                    "match_id": file_path.stem,
                    "parse_error": str(e),
                }
            )

    df = pd.DataFrame(rows)

    if "date_start" in df.columns:
        df["date_start"] = pd.to_datetime(df["date_start"], errors="coerce")
    if "date_end" in df.columns:
        df["date_end"] = pd.to_datetime(df["date_end"], errors="coerce")

    sort_cols = [c for c in ["date_start", "source_bucket", "competition_folder", "file_name"] if c in df.columns]
    df = df.sort_values(sort_cols, na_position="last").reset_index(drop=True)

    return df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Indexing files under: {RAW_ROOT}")
    df = build_index(RAW_ROOT)

    df.to_csv(OUTPUT_CSV, index=False)
    df.to_parquet(OUTPUT_PARQUET, index=False)

    print(f"Done. Wrote:")
    print(f"  - {OUTPUT_CSV}")
    print(f"  - {OUTPUT_PARQUET}")
    print(f"Rows: {len(df)}")

    if "source_bucket" in df.columns:
        print("\nSource bucket counts:")
        print(df["source_bucket"].value_counts(dropna=False))

    if "official_status" in df.columns:
        print("\nOfficial status counts:")
        print(df["official_status"].value_counts(dropna=False))

    if "competition_folder" in df.columns:
        print("\nTop competition folders:")
        print(df["competition_folder"].value_counts(dropna=False).head(20))

    if "match_type" in df.columns:
        print("\nMatch type counts:")
        print(df["match_type"].value_counts(dropna=False))


if __name__ == "__main__":
    main()