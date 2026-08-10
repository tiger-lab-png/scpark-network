# Data dictionary

Three runs are deposited. **`full_run/` carries every inferential result in the
manuscript.** The other two exist for the comparisons the paper explicitly makes
and are not interchangeable with it.

All CSV files are UTF-8 (`utf-8-sig`, for Excel compatibility). Files larger than
3 MB are stored gzipped; run `python prepare_data.py` from the repository root
once to restore them in place.

---

## `full_run/` — full silicon-carbide population (the analysis frame)

OpenAlex Topic **T10361**, 2018–2024. Cursor-paginated retrieval returned
**54,671 works** before exhausting the cursor, against a 55,125-work screening
count — a small drift consistent with routine OpenAlex updates between screening
and retrieval.

### Raw extraction

| File | Contents |
|---|---|
| `affil_full.csv` | 319,968 rows; one row per author–institution–affiliation-string combination across 54,671 unique `Paper_ID`. Columns: `Paper_ID`, `Title`, `Year`, `Cited_By_Count`, `Author_ID`, `Author_Name`, `Raw_Affiliation`, `Standardized_Institutions` (OpenAlex's algorithmic entity resolution, comma-separated for multi-institution authorships), `Institution_Countries`. Yields **8,382** distinct standardized institutions. |
| `addr_uniq_full.csv` | 80,179 rows of which 3 are null → **80,176** unique `Raw_Affiliation` strings, the input to geocoding. |

### Geocoding

| File | Contents |
|---|---|
| `geocoded.csv` | 80,176 rows, one per unique raw string, with `Latitude`/`Longitude` and the query variant that succeeded under the address-simplification cascade. **69,455 (86.6%)** carry valid coordinates. |
| `geo_cache.json` | Raw Nominatim response cache, keyed by query string. Retrieved 2026-07-21 at 1 request/second. |
| `geo_failed.txt` | Strings for which every cascade variant failed. |

### Park ground truth

| File | Contents |
|---|---|
| `parks_wikidata.csv` | 231 candidate science/technology parks from the Wikidata Query Service (instances of Q1976594 or Q1281153 carrying coordinates), with `wikidata_id`, `label`, `lat`, `lon`, and `osm_relation_id` where a P402 link exists. |
| `park_polygons.json` | Exact OSM polygon geometry, as assembled closed rings, for the parks with a linked OSM relation; keyed by Wikidata ID. |
| `park_matches.csv` | Per raw affiliation string: `distance_to_park_m`, `nearest_park_name`, `in_park_best_guess` (Method A: distance threshold combined with polygon containment under the asymmetric confidence rule). |
| `parks_prelabel_checkpoint.csv`, `wikidata_labels_cache.json` | Retrieval provenance for the registry step. |

### Method B — OSM cross-validation and infrastructure density

| File | Contents |
|---|---|
| `enriched.csv` | Per raw affiliation string: OSM `is_in()` keyword park classification (an independent cross-check on Method A), plus `univ_research_count` (2,000 m), `nearest_station_m`, `nearest_junction_m` (both right-censored at the 4,000 m search radius). |
| `overpass_cache.json` | Raw Overpass responses, keyed by coordinate; 13,817 unique coordinates across the 69,455 geocoded records. |
| `combined.csv` | Method A merged with Method B per raw affiliation string, for the cross-validation reported in Results. |
| `density_5000m.csv` | University/research-institute count within 5,000 m per unique coordinate. **August 2026 OSM retrieval — this is the vintage used in the headline specification.** |
| `density_2000m_v2.csv` | The 2,000 m covariate re-collected at the *same* date as the 5,000 m pull. Re-fitting with it reproduces IRR 1.2348 exactly, which is what establishes that the apparent radius effect was retrieval vintage rather than spatial scale. |
| `density2000_lowerbound.csv` | The earlier, thinner 2,000 m retrieval, retained so the vintage comparison can be re-executed. |
| `density5000_cache.json`, `density5000_provenance.json`, `density2000v2_cache.json`, `density2000v2_provenance.json` | Per-request Overpass provenance records: mirror used, timestamp, response status. |

### Networks

| File | Nodes | Edges | Role |
|---|---|---|---|
| `std_nodes.json` / `std_edges.json` | **7,886** | 46,595 | Entity-resolved co-affiliation network restricted to institutions with valid coordinates. One node per `Standardized_Institutions` value, carrying `degree`, `betweenness`, `community` (greedy modularity), `in_science_park`, and coordinates. **This is the primary analysis network for every inferential result.** Of the 8,382 standardized institutions, 7,886 survive the coordinate requirement; 465 of those have degree 0 and are retained. |
| `nodes.json` / `edges.json` | 69,455 | 113,330 | Naive network, one node per raw affiliation string, restricted to strings with valid coordinates (80,176 before the restriction). Used **only** as the methodological comparison in Table 2. |

Park status propagates from the affiliation-record level to the institution level
by majority vote: **187** park-affiliated institutions at 2,000 m and **504** at
5,000 m, out of 7,886. Those institution-level counts, not the record-level
matches, are the treated-group sizes in every regression.

---

## `cart_run/` — CAR-T cell-therapy cross-field probe

Same apparatus, different topic, capped sample. Supports Table 6 and the
cross-field probe section. File names and schemas mirror `full_run/`. Two
limitations stated in the manuscript apply specifically here: the sample is
capped, and registry coverage is confirmed incomplete for two of the three
motivating biotechnology clusters.

---

## `pilot_5000/` — initial 5,000-work cursor sample

The first cursor-paginated extraction: 10,404 raw affiliation strings across
4,997 works, 9,440 geocoded (90.7%), **2,663** standardized institutions of which
2,598 carry valid coordinates. Skewed toward earlier publication years by
unrandomized cursor pagination.

It is deposited because the manuscript uses it as a **designed sample-size
comparison** — the naive-to-resolved node ratio moves from 3.9:1 here to 9.6:1 in
the full population, which is what shows that entity-resolution noise does not
average out as the sample grows. It is not the analysis frame, and no inferential
result in the paper rests on it.

---

## Regenerating versus reusing

Nominatim and the public Overpass API are volunteer-run and rate-limited.
Re-running Steps 1–7 against live services takes several hours and will not
return byte-identical results, because OSM amenity coverage grows monotonically
as volunteers add features — the vintage effect the paper documents. Reuse the
deposited files unless you are extending the pipeline to a new OpenAlex topic.
