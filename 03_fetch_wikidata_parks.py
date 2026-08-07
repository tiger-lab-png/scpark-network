"""
Extraction step for the CAR-T replication dataset.

Identical to 01_fetch_openalex.py except for TOPIC_ID (T11491, CAR-T cell
therapy research) and a max_records cap of 5,000 works.

Reads:  nothing (queries the OpenAlex /works endpoint directly).
Writes: affil_full.csv and addr_uniq_full.csv, using the same filenames as
        01_fetch_openalex.py so that steps 02 and 05-09 run unchanged; run this
        variant in its own working directory so the two datasets never mix.
Set before running: MAILTO, and the publication-year window if it differs.
"""

import requests
import pandas as pd
import time

# OpenAlex, Nominatim and Overpass all ask for a contact address so they can
# reach the operator of a script that misbehaves.
MAILTO = "your-email@example.com"
BASE_URL = "https://api.openalex.org"
TOPIC_ID = "https://openalex.org/T11491"  # CAR-T cell therapy research


def fetch_works_by_topic(topic_id, year_from=2018, year_to=2024, max_records=5000):
    works = []
    cursor = "*"
    per_page = 100

    while len(works) < max_records:
        params = {
            "filter": f"topics.id:{topic_id},publication_year:{year_from}-{year_to}",
            "per-page": per_page,
            "cursor": cursor,
            "mailto": MAILTO,
        }
        resp = requests.get(f"{BASE_URL}/works", params=params, timeout=30)
        if resp.status_code != 200:
            print(f"API error {resp.status_code}: {resp.text[:200]}")
            break

        data = resp.json()
        batch = data.get("results", [])
        if not batch:
            break

        works.extend(batch)
        cursor = data.get("meta", {}).get("next_cursor")
        print(f"fetched {len(works)} works...")

        if not cursor:
            print("cursor exhausted, stopping early.")
            break
        time.sleep(0.2)

    return works[:max_records]


def extract_affiliations(works):
    rows = []
    for w in works:
        paper_id = w.get("id")
        title = w.get("title")
        year = w.get("publication_year")
        cited_by = w.get("cited_by_count")

        for authorship in w.get("authorships", []):
            author = authorship.get("author", {}) or {}
            author_id = author.get("id")
            author_name = author.get("display_name")

            # Some records expose a list of raw affiliation strings, some a
            # single one; collect both so no affiliation text is lost.
            raw_list = authorship.get("raw_affiliation_strings") or []
            raw_single = authorship.get("raw_affiliation_string")
            if raw_single and raw_single not in raw_list:
                raw_list = raw_list + [raw_single]

            institutions = authorship.get("institutions", []) or []
            inst_names = [i.get("display_name") for i in institutions if i]
            inst_countries = [i.get("country_code") for i in institutions if i]

            if not raw_list:
                raw_list = [None]

            for raw_aff in raw_list:
                rows.append({
                    "Paper_ID": paper_id,
                    "Title": title,
                    "Year": year,
                    "Cited_By_Count": cited_by,
                    "Author_ID": author_id,
                    "Author_Name": author_name,
                    "Raw_Affiliation": raw_aff,
                    "Standardized_Institutions": ", ".join(inst_names),
                    "Institution_Countries": ", ".join(filter(None, inst_countries)),
                })
    return pd.DataFrame(rows)


def build_unique_address_list(df):
    unique_addresses = (
        df.dropna(subset=["Raw_Affiliation"])["Raw_Affiliation"]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return unique_addresses.to_frame(name="Raw_Affiliation")


if __name__ == "__main__":
    works_data = fetch_works_by_topic(TOPIC_ID, year_from=2018, year_to=2024, max_records=5000)
    df_affiliations = extract_affiliations(works_data)

    # Deliberately the same filenames as 01_fetch_openalex.py: the downstream
    # scripts are reused verbatim, so keep this run in a separate directory.
    df_affiliations.to_csv("affil_full.csv", index=False, encoding="utf-8-sig")

    df_unique_addr = build_unique_address_list(df_affiliations)
    df_unique_addr.to_csv("addr_uniq_full.csv", index=False, encoding="utf-8-sig")

    print(f"{len(df_affiliations)} author-affiliation records, "
          f"{len(df_unique_addr)} unique addresses to geocode.")
    print(f"Distinct works: {df_affiliations['Paper_ID'].nunique()} "
          f"(OpenAlex keeps adding and revising records, so a re-extraction is "
          f"not guaranteed to reproduce an earlier sample exactly).")
