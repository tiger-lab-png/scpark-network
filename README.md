"""
Build the science-park registry from Wikidata.

Reads:  nothing (queries the Wikidata Query Service and the MediaWiki API).
Writes: parks_wikidata.csv  one row per park with wikidata_id, lat, lon,
                            osm_relation_id (where present), name and country.
        Intermediate checkpoints (parks_prelabel_checkpoint.csv,
        wikidata_labels_cache.json) let an interrupted run resume.
Set before running: MAILTO, and CANDIDATE_CLASS_QIDS if the registry should
cover further park classes. A copy of the resulting registry ships in data/, so
this step only needs to be re-run to refresh it.
The script refuses to overwrite parks_wikidata.csv when the self-check fails.
"""

import json
import os
import time

import pandas as pd
import requests

WIKIDATA_SPARQL_URL = "https://query.wikidata.org/sparql"
WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"

# OpenAlex, Nominatim and Overpass all ask for a contact address so they can
# reach the operator of a script that misbehaves; Wikidata likewise requires an
# identifiable User-Agent.
MAILTO = "your-email@example.com"

HEADERS = {
    "User-Agent": f"micro-geo-innovation-mapper/0.1 ({MAILTO})",
    "Accept": "application/sparql-results+json",
}

# Seed classes for "science park" / "technology park". wdt:P279* expands them to
# their subclasses, which covers naming variants (research park, technopark, ...).
CANDIDATE_CLASS_QIDS = [
    "Q1976594",  # science park
    "Q1281153",  # technology park
]

SUBCLASS_QUERY = """
SELECT DISTINCT ?class WHERE {{
  VALUES ?seed {{ {seeds} }}
  ?class wdt:P279* ?seed .
}}
""".format(seeds=" ".join(f"wd:{q}" for q in CANDIDATE_CLASS_QIDS))


def sparql_request(query, timeout=60, max_retries=3, use_post=False):
    """Shared SPARQL helper; use POST for long queries that would blow the URL limit."""
    for attempt in range(max_retries):
        try:
            if use_post:
                resp = requests.post(
                    WIKIDATA_SPARQL_URL,
                    data={"query": query, "format": "json"},
                    headers=HEADERS,
                    timeout=timeout,
                )
            else:
                resp = requests.get(
                    WIKIDATA_SPARQL_URL,
                    params={"query": query, "format": "json"},
                    headers=HEADERS,
                    timeout=timeout,
                )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            wait = 2 ** attempt
            print(f"[wikidata retry {attempt + 1}/{max_retries}] {e}")
            time.sleep(wait)
    raise RuntimeError("Wikidata SPARQL query failed after repeated retries.")


def fetch_subclasses():
    """
    Stage 1: expand the seed classes on their own. Keeping the P279* property
    path in a query with no other joins is what makes it fast enough for the
    public endpoint.
    """
    data = sparql_request(SUBCLASS_QUERY, timeout=60)
    class_qids = [
        b["class"]["value"].rsplit("/", 1)[-1]
        for b in data["results"]["bindings"]
    ]
    class_qids = sorted(set(class_qids))
    print(f"expanded to {len(class_qids)} class QIDs")
    if len(class_qids) > 500:
        print("  warning: unusually many classes, which can mean a seed QID sits "
              "under a very general parent; spot-check a few class QIDs.")
    return class_qids


def fetch_instances_for_classes(class_qids, timeout=90):
    """
    Stage 2: fetch the park items with VALUES + a direct wdt:P31 comparison
    (indexed) instead of a property path, and POST because the QID list is long.
    """
    query = """
    SELECT DISTINCT ?item ?coord ?osmRelation ?country WHERE {{
      VALUES ?class {{ {classes} }}
      ?item wdt:P31 ?class .
      ?item wdt:P625 ?coord .
      OPTIONAL {{ ?item wdt:P402 ?osmRelation . }}
      OPTIONAL {{ ?item wdt:P17 ?country . }}
    }}
    """.format(classes=" ".join(f"wd:{q}" for q in class_qids))
    return sparql_request(query, timeout=timeout, use_post=True)


LABEL_CACHE_FILE = "wikidata_labels_cache.json"


def _load_label_cache():
    if os.path.exists(LABEL_CACHE_FILE):
        with open(LABEL_CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_label_cache(cache):
    # atomic replace, as in 02_geocode_and_enrich.py
    tmp_path = LABEL_CACHE_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, LABEL_CACHE_FILE)


def fetch_labels_batch(qids, batch_size=50, max_retries=3):
    """
    Stage 3: resolve labels through the MediaWiki wbgetentities API instead of
    SPARQL's label service, which is markedly faster and more reliable. Labels
    are cached, so a failure part-way through does not discard earlier work.
    """
    cache = _load_label_cache()
    qids = list(dict.fromkeys(qids))  # deduplicate, preserving order
    missing = [q for q in qids if q not in cache]

    if missing:
        print(f"{len(cache)} labels cached, {len(missing)} still to fetch...")
        for i in range(0, len(missing), batch_size):
            chunk = missing[i:i + batch_size]
            entities = None
            for attempt in range(max_retries):
                try:
                    resp = requests.get(
                        WIKIDATA_API_URL,
                        params={
                            "action": "wbgetentities",
                            "ids": "|".join(chunk),
                            "props": "labels",
                            # prefer English, then fall back to any of these
                            # languages rather than displaying a bare QID
                            "languages": "en|de|fr|ja|zh",
                            "format": "json",
                        },
                        headers=HEADERS,
                        timeout=30,
                    )
                    resp.raise_for_status()
                    entities = resp.json().get("entities", {})
                    break
                except Exception as e:
                    wait = 2 ** attempt
                    print(f"  [label retry {attempt + 1}/{max_retries}] {e}")
                    time.sleep(wait)

            if entities is None:
                raise RuntimeError(
                    f"label lookup failed after {max_retries} retries (items "
                    f"{i}~{i + len(chunk)}). The {len(cache)} labels resolved so "
                    f"far are cached in {LABEL_CACHE_FILE}; re-running this "
                    f"script fetches only the remainder."
                )

            for qid, ent in entities.items():
                ent_labels = ent.get("labels", {})
                label = None
                for lang in ("en", "de", "fr", "ja", "zh"):
                    if lang in ent_labels:
                        label = ent_labels[lang]["value"]
                        break
                cache[qid] = label or qid
            _save_label_cache(cache)
            print(f"  labels {min(i + batch_size, len(missing))}/{len(missing)} "
                  f"({len(cache)} cached in total)")

    return {q: cache.get(q, q) for q in qids}


PRELABEL_CHECKPOINT_FILE = "parks_prelabel_checkpoint.csv"
PRELABEL_CHECKPOINT_META_FILE = "parks_prelabel_checkpoint.meta.json"


def _checkpoint_is_valid():
    """
    A checkpoint may only be reused when it was produced from the current
    CANDIDATE_CLASS_QIDS; otherwise editing that list after a failed self-check
    would have no effect.
    """
    if not (os.path.exists(PRELABEL_CHECKPOINT_FILE)
            and os.path.exists(PRELABEL_CHECKPOINT_META_FILE)):
        return False
    with open(PRELABEL_CHECKPOINT_META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return meta.get("candidate_class_qids") == sorted(CANDIDATE_CLASS_QIDS)


def _save_checkpoint(df):
    df.to_csv(PRELABEL_CHECKPOINT_FILE, index=False, encoding="utf-8-sig")
    with open(PRELABEL_CHECKPOINT_META_FILE, "w", encoding="utf-8") as f:
        json.dump({"candidate_class_qids": sorted(CANDIDATE_CLASS_QIDS)}, f)


def fetch_wikidata_parks():
    """
    Three stages: expand subclasses, fetch the items, then resolve labels. The
    first two stages are checkpointed so a label failure does not force the
    SPARQL queries to be repeated.
    """
    if _checkpoint_is_valid():
        print(f"reusing checkpoint {PRELABEL_CHECKPOINT_FILE} (matches the current "
              f"CANDIDATE_CLASS_QIDS); skipping the SPARQL stages...")
        df = pd.read_csv(PRELABEL_CHECKPOINT_FILE)
    else:
        class_qids = fetch_subclasses()
        data = fetch_instances_for_classes(class_qids)

        rows = []
        for b in data["results"]["bindings"]:
            coord_str = b.get("coord", {}).get("value", "")
            # Wikidata coordinates are WKT: "Point(lon lat)" -- longitude first
            lat, lon = None, None
            if coord_str.startswith("Point("):
                try:
                    lon_str, lat_str = coord_str[6:-1].split(" ")
                    lat, lon = float(lat_str), float(lon_str)
                except ValueError:
                    item_id = b.get("item", {}).get("value", "?")
                    print(f"  unparseable coordinate, skipping: {item_id} -> {coord_str!r}")

            country_uri = b.get("country", {}).get("value")
            country_qid = country_uri.rsplit("/", 1)[-1] if country_uri else None

            rows.append({
                "wikidata_id": b["item"]["value"].rsplit("/", 1)[-1],
                "lat": lat,
                "lon": lon,
                "osm_relation_id": b.get("osmRelation", {}).get("value"),
                "country_qid": country_qid,
            })

        df = pd.DataFrame(rows).dropna(subset=["lat", "lon"])
        df = df.drop_duplicates(subset=["wikidata_id"])
        _save_checkpoint(df)
        print(f"checkpoint written to {PRELABEL_CHECKPOINT_FILE} ({len(df)} rows, labels pending)")

    country_qids = set(df["country_qid"].dropna().unique())
    print(f"{len(df)} park items retrieved, resolving labels...")
    all_qids = list(df["wikidata_id"]) + list(country_qids)
    labels = fetch_labels_batch(all_qids)

    df["name"] = df["wikidata_id"].map(labels)
    df["country"] = df["country_qid"].map(labels)
    df = df.drop(columns=["country_qid"])
    return df


def self_check(df):
    """
    Known-case validation against Hsinchu Science Park (Q717461). This registry
    is the ground truth for steps 05-07, so silently writing an unvalidated list
    would be worse than failing loudly; the caller decides what to do with False.
    """
    hit = df[df["wikidata_id"] == "Q717461"]
    if len(hit) > 0:
        print(f"self-check passed: Hsinchu Science Park found -> {hit.iloc[0].to_dict()}")
        return True
    else:
        print("self-check FAILED: Hsinchu Science Park (Q717461) is missing, so "
              "CANDIDATE_CLASS_QIDS needs adjusting and this list cannot be "
              "trusted. Check https://www.wikidata.org/wiki/Q717461 for the P31 "
              "class it is actually filed under and add that QID.")
        return False


if __name__ == "__main__":
    print("querying Wikidata for science and technology parks...")
    df_parks = fetch_wikidata_parks()
    print(f"{len(df_parks)} park entries with coordinates")

    has_osm = df_parks["osm_relation_id"].notna().sum()
    print(f"{has_osm} of them link an OSM relation ID (usable for exact polygons in step 04)")

    if self_check(df_parks):
        df_parks.to_csv("parks_wikidata.csv", index=False, encoding="utf-8-sig")
        print("written to parks_wikidata.csv")
    else:
        df_parks.to_csv("parks_wikidata_UNVERIFIED.csv", index=False, encoding="utf-8-sig")
        print("Self-check failed, so parks_wikidata.csv was NOT written and steps "
              "05-07 cannot pick up a faulty ground truth. The result is in "
              "parks_wikidata_UNVERIFIED.csv for inspection; fix "
              "CANDIDATE_CLASS_QIDS and re-run.")
        raise SystemExit(1)
