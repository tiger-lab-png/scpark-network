"""
Merge the two park classifications into one table.

Reads:  enriched.csv     (step 02: OSM/Overpass verdict plus the density and
                          accessibility covariates)
        park_matches.csv (steps 05-06: registry verdict, nearest park, distance)
Writes: combined.csv -- enriched.csv left-joined on Raw_Affiliation, with
        *_osm columns for the OSM verdict, *_gt columns for the registry
        verdict, and `agree` where both are available.

Raw_Affiliation is unique in both inputs (step 01 deduplicates the address
list), so the join cannot multiply rows. `agree` compares only the inside/outside
verdicts, not the park names, which legitimately differ between sources.
"""

import pandas as pd

if __name__ == "__main__":
    df_enriched = pd.read_csv("enriched.csv")
    df_gt = pd.read_csv("park_matches.csv")

    df_enriched = df_enriched.rename(columns={
        "in_science_park": "in_science_park_osm",
        "park_name": "park_name_osm",
        "method_used": "method_used_osm",
    })

    df_gt_slim = df_gt[[
        "Raw_Affiliation", "nearest_park_name", "nearest_park_wikidata_id",
        "distance_to_park_m", "in_science_park_gt",
    ]].rename(columns={"nearest_park_name": "park_name_gt"})

    df_combined = df_enriched.merge(df_gt_slim, on="Raw_Affiliation", how="left")

    both_known = df_combined["in_science_park_osm"].notna() & df_combined["in_science_park_gt"].notna()
    df_combined["agree"] = None
    df_combined.loc[both_known, "agree"] = (
        df_combined.loc[both_known, "in_science_park_osm"]
        == df_combined.loc[both_known, "in_science_park_gt"]
    )

    df_combined.to_csv("combined.csv", index=False, encoding="utf-8-sig")

    n_both = both_known.sum()
    n_agree = (df_combined.loc[both_known, "agree"] == True).sum()
    n_gt_true = (df_combined["in_science_park_gt"] == True).sum()
    n_osm_true = (df_combined["in_science_park_osm"] == True).sum()
    n_both_true = ((df_combined["in_science_park_gt"] == True)
                   & (df_combined["in_science_park_osm"] == True)).sum()
    n_gt_only = ((df_combined["in_science_park_gt"] == True)
                 & (df_combined["in_science_park_osm"] == False)).sum()
    n_osm_only = ((df_combined["in_science_park_osm"] == True)
                  & (df_combined["in_science_park_gt"] == False)).sum()

    print(f"combined.csv written: {len(df_combined)} rows")
    print(f"\nrows with a verdict from both methods: {n_both}")
    print(f"same verdict: {n_agree} ({n_agree / n_both * 100:.1f}%)" if n_both else "")
    print(f"\nmethod A (Wikidata registry) inside a park: {n_gt_true}")
    print(f"method B (Overpass/OSM) inside a park: {n_osm_true}")
    print(f"both: {n_both_true}")
    print(f"method A only (in the registry, not mapped in OSM): {n_gt_only}")
    print(f"method B only (mapped in OSM, absent from the registry): {n_osm_only}")
    print("\nThe agreement rate is itself a reportable figure: disagreements are "
          "typically newer or smaller parks that OSM has mapped but the registry "
          "has not yet listed, or registry parks whose boundary no OSM "
          "contributor has drawn.")
