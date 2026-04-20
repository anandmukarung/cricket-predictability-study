from __future__ import annotations

import json
import os
import random
import re
import time
from pathlib import Path

import mwparserfromhell
import pandas as pd
import requests
from requests.exceptions import ChunkedEncodingError, ConnectionError, ReadTimeout
from tqdm import tqdm


PLAYERS_SEED_CSV = Path("data/interim/players_seed_v2.csv")
OUTPUT_CSV = Path("data/raw/external/t20s_players_wikipedia_enriched_v2.csv")

WDQS_URL = "https://query.wikidata.org/sparql"
WIKI_API = "https://en.wikipedia.org/w/api.php"

MAP_CACHE = Path("data/interim/cache_cricinfo_to_enwiki_v2.json")
PAGE_CACHE = Path("data/interim/cache_enwiki_wikitext_v2.json")
ROWS_CACHE = Path("data/interim/cache_enriched_rows_v2.jsonl")

HEADERS_WDQS = {
    "Accept": "application/sparql-results+json",
    "User-Agent": "t20s-wikipedia-enricher/2.0 (contact: anandmukarung@gmail.com)",
}
HEADERS_WIKI = {
    "User-Agent": "t20s-wikipedia-enricher/2.0 (contact: anandmukarung@gmail.com)",
}

EXPECTED_FIELDS = [
    "batting_style",
    "bowling_style",
    "playing_role",
    "infobox_template",
    "wikidata_item",
    "enwiki_title",
    "enwiki_rev_id",
    "enwiki_rev_timestamp",
]

SESSION = requests.Session()
SESSION.headers.update(HEADERS_WIKI)


def load_json(path: Path, default):
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return default


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


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


def load_seed() -> pd.DataFrame:
    if not PLAYERS_SEED_CSV.exists():
        raise FileNotFoundError(f"Missing {PLAYERS_SEED_CSV}")

    df = pd.read_csv(PLAYERS_SEED_CSV).copy()

    required = {"player_name", "cricsheet_id"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"players_seed_v2.csv missing required columns: {missing}")

    if "cricinfo_id" not in df.columns:
        df["cricinfo_id"] = None
    if "playing_role" not in df.columns:
        df["playing_role"] = None
    if "batting_style" not in df.columns:
        df["batting_style"] = None
    if "bowling_style" not in df.columns:
        df["bowling_style"] = None

    df["player_name_norm"] = normalize_name(df["player_name"])
    df["cricsheet_id_norm"] = normalize_id(df["cricsheet_id"])
    df["cricinfo_id_norm"] = normalize_id(df["cricinfo_id"])

    df = (
        df.sort_values(["player_name_norm"])
        .drop_duplicates(subset=["cricsheet_id_norm", "player_name_norm"], keep="first")
        .reset_index(drop=True)
    )

    return df


def needs_enrichment_mask(df: pd.DataFrame) -> pd.Series:
    role_missing = df["playing_role"].isna() | (df["playing_role"].astype(str).str.strip() == "")
    batting_missing = df["batting_style"].isna() | (df["batting_style"].astype(str).str.strip() == "")
    bowling_missing = df["bowling_style"].isna() | (df["bowling_style"].astype(str).str.strip() == "")
    return role_missing | batting_missing | bowling_missing


def wdqs_map_batch(cricinfo_ids):
    vals = " ".join(f'"{str(int(v))}"' for v in cricinfo_ids)
    query = f"""
    SELECT ?cid ?item ?enwiki WHERE {{
      VALUES ?cid {{ {vals} }}
      ?item wdt:P2697 ?cid .
      OPTIONAL {{
        ?enwiki schema:about ?item ;
               schema:isPartOf <https://en.wikipedia.org/> .
      }}
    }}
    """

    for attempt in range(10):
        r = requests.get(WDQS_URL, params={"query": query}, headers=HEADERS_WDQS, timeout=90)

        if r.status_code == 200:
            data = r.json()
            out = {}
            for b in data.get("results", {}).get("bindings", []):
                cid = b.get("cid", {}).get("value")
                item = b.get("item", {}).get("value")
                enwiki_url = b.get("enwiki", {}).get("value")
                title = None
                if enwiki_url and "en.wikipedia.org/wiki/" in enwiki_url:
                    title = enwiki_url.split("/wiki/", 1)[1].replace("_", " ")
                if cid and cid.isdigit():
                    out[int(cid)] = {"wikidata_item": item, "enwiki_title": title}
            return out

        if r.status_code in (429, 500, 502, 503, 504):
            wait = min(60, (2**attempt)) + random.uniform(0.5, 2.0)
            retry_after = r.headers.get("Retry-After")
            if retry_after and retry_after.isdigit():
                wait = max(wait, int(retry_after))
            time.sleep(wait)
            continue

        r.raise_for_status()

    return {}


def wiki_fetch_wikitext_and_rev(title, max_retries=15):
    params = {
        "action": "query",
        "prop": "revisions",
        "rvprop": "ids|timestamp|content",
        "rvslots": "main",
        "format": "json",
        "formatversion": 2,
        "titles": title,
    }

    for attempt in range(max_retries):
        try:
            r = SESSION.get(WIKI_API, params=params, timeout=60)

            if r.status_code == 200:
                js = r.json()
                pages = js.get("query", {}).get("pages", [])
                if not pages or pages[0].get("missing"):
                    return None, None, None
                revs = pages[0].get("revisions", [])
                if not revs:
                    return None, None, None
                rev = revs[0]
                return (
                    rev.get("slots", {}).get("main", {}).get("content"),
                    rev.get("revid"),
                    rev.get("timestamp"),
                )

            if r.status_code in (429, 500, 502, 503, 504):
                wait = min(180, (2**attempt)) + random.uniform(0.5, 2.5)
                retry_after = r.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    wait = max(wait, int(retry_after))
                time.sleep(wait)
                continue

            r.raise_for_status()

        except (ConnectionError, ReadTimeout, ChunkedEncodingError):
            wait = min(120, (2**attempt)) + random.uniform(0.5, 2.5)
            time.sleep(wait)
            continue
        except Exception:
            wait = min(60, (2**attempt)) + random.uniform(0.5, 2.0)
            time.sleep(wait)
            continue

    return None, None, None


def clean_value(v):
    if v is None:
        return None
    v = str(v).strip()
    if not v:
        return None
    v = re.sub(r"<ref[^>]*>.*?</ref>", "", v, flags=re.I | re.S)
    v = re.sub(r"<ref[^/>]*/\s*>", "", v, flags=re.I)
    v = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", v)
    v = re.sub(r"\[https?://[^\s\]]+\s*([^\]]*)\]", r"\1", v)
    v = re.sub(r"\{\{.*?\}\}", "", v)
    v = re.sub(r"\s+", " ", v).strip(" -–—\t\r\n")
    return v or None


def extract_infobox_fields(wikitext):
    if not wikitext:
        return {k: None for k in EXPECTED_FIELDS}

    code = mwparserfromhell.parse(wikitext)
    templates = code.filter_templates(recursive=True)

    target_names = {
        "infobox cricketer",
        "infobox cricket player",
        "infobox cricketer biography",
        "infobox sportsperson",
    }

    chosen = None
    for t in templates:
        name = str(t.name).strip().lower()
        if name in target_names:
            chosen = t
            break

    if chosen is None:
        for t in templates:
            params = {str(p.name).strip().lower() for p in t.params}
            if "bowling" in params or "bowling style" in params or "batting" in params:
                chosen = t
                break

    if chosen is None:
        return {k: None for k in EXPECTED_FIELDS}

    def get_param(*names):
        for n in names:
            if chosen.has(n):
                return clean_value(chosen.get(n).value)
        return None

    return {
        "batting_style": get_param("batting", "batting style", "batting_style"),
        "bowling_style": get_param("bowling", "bowling style", "bowling_style"),
        "playing_role": get_param("role", "playing role", "playing_role"),
        "infobox_template": str(chosen.name).strip(),
        "wikidata_item": None,
        "enwiki_title": None,
        "enwiki_rev_id": None,
        "enwiki_rev_timestamp": None,
    }


def load_existing_rows_cache() -> pd.DataFrame:
    rows = []
    if ROWS_CACHE.exists():
        with ROWS_CACHE.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass

    if not rows:
        return pd.DataFrame(columns=["cricinfo_id", *EXPECTED_FIELDS])

    df = pd.DataFrame(rows).drop_duplicates(subset=["cricinfo_id"], keep="last")
    return df


def main():
    seed = load_seed()

    todo = seed.loc[needs_enrichment_mask(seed)].copy()
    todo = todo.loc[todo["cricinfo_id_norm"] != ""].copy()

    todo["cricinfo_id"] = pd.to_numeric(todo["cricinfo_id"], errors="coerce")
    todo = todo.loc[todo["cricinfo_id"].notna()].copy()
    todo["cricinfo_id"] = todo["cricinfo_id"].astype(int)

    print(f"Seed rows: {len(seed)}")
    print(f"Players needing role/style enrichment: {needs_enrichment_mask(seed).sum()}")
    print(f"Players needing enrichment with cricinfo_id available: {len(todo)}")

    ids = sorted(todo["cricinfo_id"].drop_duplicates().tolist())

    map_cache = load_json(MAP_CACHE, {})
    page_cache = load_json(PAGE_CACHE, {})

    missing_map = [i for i in ids if str(i) not in map_cache]
    if missing_map:
        for batch in tqdm(list(chunked(missing_map, 200)), desc="Wikidata mapping"):
            got = wdqs_map_batch(batch)
            for k, v in got.items():
                map_cache[str(k)] = v
            save_json(MAP_CACHE, map_cache)
            time.sleep(0.4)

    mapping_rows = []
    for cid in ids:
        m = map_cache.get(str(cid))
        if m and m.get("enwiki_title"):
            mapping_rows.append((cid, m.get("wikidata_item"), m.get("enwiki_title")))

    existing_cache_df = load_existing_rows_cache()
    done_ids = set()
    if "cricinfo_id" in existing_cache_df.columns:
        done_ids = set(pd.to_numeric(existing_cache_df["cricinfo_id"], errors="coerce").dropna().astype(int).tolist())

    print(f"Mapped to enwiki titles: {len(mapping_rows)}")

    for cid, qid, title in tqdm(mapping_rows, desc="Wikipedia pages"):
        if cid in done_ids:
            continue

        if title in page_cache:
            wikitext = page_cache[title].get("wikitext")
            rev_id = page_cache[title].get("rev_id")
            ts = page_cache[title].get("ts")
        else:
            wikitext, rev_id, ts = wiki_fetch_wikitext_and_rev(title)
            page_cache[title] = {"wikitext": wikitext, "rev_id": rev_id, "ts": ts}
            if len(page_cache) % 50 == 0:
                save_json(PAGE_CACHE, page_cache)

        fields = extract_infobox_fields(wikitext)
        fields["wikidata_item"] = qid
        fields["enwiki_title"] = title
        fields["enwiki_rev_id"] = rev_id
        fields["enwiki_rev_timestamp"] = ts

        row = {
            "cricinfo_id": int(cid),
            **fields,
        }
        append_jsonl(ROWS_CACHE, row)
        done_ids.add(cid)
        time.sleep(0.9 + random.uniform(0.0, 0.4))

    save_json(PAGE_CACHE, page_cache)

    enriched_rows_df = load_existing_rows_cache()

    out = seed.copy()

    merge_cols = ["cricinfo_id", *EXPECTED_FIELDS]
    available_merge_cols = [c for c in merge_cols if c in enriched_rows_df.columns]

    out = out.merge(
        enriched_rows_df[available_merge_cols],
        on="cricinfo_id",
        how="left",
        suffixes=("", "_wiki"),
    )

    for col in EXPECTED_FIELDS:
        wiki_col = f"{col}_wiki"
        if wiki_col in out.columns:
            out[col] = out[col].combine_first(out[wiki_col])

    drop_cols = [c for c in out.columns if c.endswith("_wiki")]
    if drop_cols:
        out = out.drop(columns=drop_cols)

    output_cols = [
        "player_name",
        "cricsheet_id",
        "cricinfo_id",
        "playing_role",
        "batting_style",
        "bowling_style",
        "wikidata_item",
        "enwiki_title",
        "enwiki_rev_id",
        "enwiki_rev_timestamp",
        "infobox_template",
    ]
    output_cols = [c for c in output_cols if c in out.columns]
    out = out[output_cols].copy()

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)

    print("Wrote:", OUTPUT_CSV)
    print("Coverage summary:")
    print(f"  rows: {len(out)}")
    print(f"  with cricsheet_id: {out['cricsheet_id'].notna().sum()}")
    print(f"  with cricinfo_id: {out['cricinfo_id'].notna().sum()}")
    print(f"  with playing_role: {out['playing_role'].notna().sum()}")
    print(f"  with batting_style: {out['batting_style'].notna().sum()}")
    print(f"  with bowling_style: {out['bowling_style'].notna().sum()}")


if __name__ == "__main__":
    main()