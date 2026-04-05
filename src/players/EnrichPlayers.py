import os, time, json, random, re
import pandas as pd
import requests
import mwparserfromhell
from tqdm import tqdm
from requests.exceptions import ConnectionError, ReadTimeout, ChunkedEncodingError


INPUT_CSV = "t20s_players_cricinfo_join.csv"
OUTPUT_CSV = "t20s_players_wikipedia_enriched.csv"

WDQS_URL = "https://query.wikidata.org/sparql"
WIKI_API = "https://en.wikipedia.org/w/api.php"

# Resumable caches
MAP_CACHE = "cache_cricinfo_to_enwiki.json"      # cricinfo_id -> {wikidata_item, enwiki_title}
PAGE_CACHE = "cache_enwiki_wikitext.json"        # enwiki_title -> {rev_id, ts, wikitext}
ROWS_CACHE = "cache_enriched_rows.jsonl"         # JSON lines of enriched rows (append-only)

HEADERS_WDQS = {
    "Accept": "application/sparql-results+json",
    "User-Agent": "t20s-wikipedia-enricher/1.2 (contact: anandrai)"
}
HEADERS_WIKI = {
    "User-Agent": "t20s-wikipedia-enricher/1.2 (contact: anandrai)"
}

EXPECTED_FIELDS = ["batting_style", "bowling_style", "playing_role", "infobox_template"]


SESSION = requests.Session()
SESSION.headers.update(HEADERS_WIKI)

def load_json(path, default):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return default

def save_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)

def append_jsonl(path, row):
    with open(path, "a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

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
            wait = min(60, (2 ** attempt)) + random.uniform(0.5, 2.0)
            ra = r.headers.get("Retry-After")
            if ra and ra.isdigit():
                wait = max(wait, int(ra))
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
        "titles": title
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

            # retry-able HTTP codes
            if r.status_code in (429, 500, 502, 503, 504):
                wait = min(180, (2 ** attempt)) + random.uniform(0.5, 2.5)
                ra = r.headers.get("Retry-After")
                if ra and ra.isdigit():
                    wait = max(wait, int(ra))
                time.sleep(wait)
                continue

            r.raise_for_status()

        except (ConnectionError, ReadTimeout, ChunkedEncodingError) as e:
            # network wobble: back off and retry
            wait = min(120, (2 ** attempt)) + random.uniform(0.5, 2.5)
            time.sleep(wait)
            continue
        except Exception:
            # any other unexpected error: small backoff then keep trying
            wait = min(60, (2 ** attempt)) + random.uniform(0.5, 2.0)
            time.sleep(wait)
            continue

    return None, None, None

def clean_value(v):
    if v is None:
        return None
    v = str(v).strip()
    if not v:
        return None
    v = re.sub(r"<ref[^>]*>.*?</ref>", "", v, flags=re.I|re.S)
    v = re.sub(r"<ref[^/>]*/\s*>", "", v, flags=re.I)
    v = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", v)  # [[A|B]] -> B
    v = re.sub(r"\[https?://[^\s\]]+\s*([^\]]*)\]", r"\1", v)  # [url label] -> label
    v = re.sub(r"\{\{.*?\}\}", "", v)  # drop nested templates conservatively
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
    }

def main():
    df = pd.read_csv(INPUT_CSV)
    ids = df["cricinfo_id"].dropna().astype(int).drop_duplicates().tolist()

    # Load caches
    map_cache = load_json(MAP_CACHE, {})          # int -> dict
    page_cache = load_json(PAGE_CACHE, {})        # title -> dict

    # --- Step 1: Ensure mapping cache has enwiki titles ---
    missing_map = [i for i in ids if str(i) not in map_cache]
    if missing_map:
        for batch in tqdm(list(chunked(missing_map, 200)), desc="Wikidata mapping"):
            got = wdqs_map_batch(batch)
            for k, v in got.items():
                map_cache[str(k)] = v
            save_json(MAP_CACHE, map_cache)
            time.sleep(0.4)  # WDQS politeness

    # Build list of titles to fetch
    mapping_rows = []
    for cid in ids:
        m = map_cache.get(str(cid))
        if m and m.get("enwiki_title"):
            mapping_rows.append((cid, m.get("wikidata_item"), m.get("enwiki_title")))

    # --- Step 2: Resume logic: see what we already enriched ---
    done_titles = set()
    if os.path.exists(ROWS_CACHE):
        with open(ROWS_CACHE, "r") as f:
            for line in f:
                try:
                    done_titles.add(json.loads(line).get("enwiki_title"))
                except Exception:
                    pass

    # --- Step 3: Fetch + parse pages slowly ---
    total = len(mapping_rows)
    print(f"Total mapped enwiki titles: {total}. Already done: {len(done_titles)}")

    for (cid, qid, title) in tqdm(mapping_rows, desc="Wikipedia pages"):
        if title in done_titles:
            continue

        # cache lookup
        if title in page_cache:
            wikitext = page_cache[title].get("wikitext")
            rev_id = page_cache[title].get("rev_id")
            ts = page_cache[title].get("ts")
        else:
            wikitext, rev_id, ts = wiki_fetch_wikitext_and_rev(title)
            page_cache[title] = {"wikitext": wikitext, "rev_id": rev_id, "ts": ts}
            # save page cache periodically
            if len(page_cache) % 50 == 0:
                save_json(PAGE_CACHE, page_cache)

        fields = extract_infobox_fields(wikitext)

        row = {
            "cricinfo_id": int(cid),
            "wikidata_item": qid,
            "enwiki_title": title,
            "enwiki_rev_id": rev_id,
            "enwiki_rev_timestamp": ts,
            **fields
        }
        append_jsonl(ROWS_CACHE, row)
        done_titles.add(title)

        # Slow down to avoid 429: aim ~1 req/sec average (plus backoff on 429)
        time.sleep(0.9 + random.uniform(0.0, 0.4))

    # Final save cache
    save_json(PAGE_CACHE, page_cache)

    # --- Step 4: Build final CSV output ---
    enriched_rows = []
    if os.path.exists(ROWS_CACHE):
        with open(ROWS_CACHE, "r") as f:
            for line in f:
                try:
                    enriched_rows.append(json.loads(line))
                except Exception:
                    pass

    enriched = pd.DataFrame(enriched_rows).drop_duplicates(subset=["cricinfo_id"], keep="last")
    out = df.merge(
        enriched[[
            "cricinfo_id","wikidata_item","enwiki_title",
            "enwiki_rev_id","enwiki_rev_timestamp",
            "batting_style","bowling_style","playing_role","infobox_template"
        ]],
        on="cricinfo_id",
        how="left"
    )

    out.to_csv(OUTPUT_CSV, index=False)
    print("Wrote:", OUTPUT_CSV)
    print("Mapped to enwiki titles:", out["enwiki_title"].notna().mean())
    print("Bowling style coverage:", out["bowling_style"].notna().mean())
    print("Batting style coverage:", out["batting_style"].notna().mean())

if __name__ == "__main__":
    main()