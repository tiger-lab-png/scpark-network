# Data dictionary

All files are UTF-8 (CSV files use `utf-8-sig` for Excel compatibility). Row counts reflect the
silicon-carbide semiconductor (Topic T10361) case study run described in the paper.

## Raw extraction

**`affil.csv`** (10,404 rows within `Paper_ID`; one row per author-institution-affiliation-string
combination, so paper count < row count)
- `Paper_ID`, `Title`, `Year`, `Cited_By_Count` — OpenAlex work metadata
- `Author_ID`, `Author_Name` — OpenAlex author metadata
- `Raw_Affiliation` — raw, unresolved affiliation string as it appears in the source publication
- `Standardized_Institutions` — OpenAlex's own algorithmic entity resolution (comma-separated if
  an author lists multiple institutions); this is the field used to build the entity-resolved
  network (Steps 8)
- `Institution_Countries` — ISO country codes associated with the standardized institutions

**`addr_uniq.csv`** — deduplicated `Raw_Affiliation` strings (10,404 unique values), the input to
Nominatim geocoding.

## Geocoding

**`geocoded.csv`** — one row per unique raw affiliation string with `Latitude`/`Longitude` (9,440 of
10,404 successfully geocoded, 90.7%) plus the specific query string variant that succeeded (see
paper Methodology, "address-simplification cascade").

**`geo_cache.json`** — raw Nominatim response cache, keyed by query string. Included for
reproducibility / to avoid re-hitting the rate-limited (1 req/sec) public Nominatim endpoint.

## Park ground truth

**`parks_wikidata.csv`** — 231 candidate science/technology parks retrieved from the Wikidata Query
Service (instances of Q1976594 "science park" or Q1281153 "technology park" with coordinate data),
including `wikidata_id`, `label`, `lat`, `lon`, and `osm_relation_id` where a P402 link exists.

**`park_polygons.json`** — exact OSM polygon geometry (as assembled closed rings) for the 10 parks
with a linked OSM relation, keyed by Wikidata ID.

**`park_matches.csv`** — per raw affiliation string: `distance_to_park_m` (haversine distance to
nearest Wikidata park), `nearest_park_name`, and `in_park_best_guess` (combined
distance-threshold + polygon-containment classification, the primary Method A output).

## Method B (OpenStreetMap cross-validation + infrastructure density)

**`enriched.csv`** — per raw affiliation string: OSM `is_in()` keyword-matched park classification
(independent cross-check on Method A) plus three density covariates: `univ_research_count`
(tagged universities/research institutes within 2,000 m), `nearest_station_m`, `nearest_junction_m`
(distance to nearest railway station / motorway junction, capped at the 4,000 m search radius when
none is found — see paper Limitations on right-censoring).

**`overpass_cache.json`** — raw Overpass API response cache, keyed by coordinate. Included to avoid
re-hitting rate-limited public Overpass mirrors.

**`combined.csv`** — merges `park_matches.csv` (Method A) and `enriched.csv` (Method B) per raw
affiliation string for direct cross-validation comparison (see paper Results, "Cross-Validation
Against Independently Tagged OSM Park Boundaries").

## Networks

**`std_nodes.json` / `std_edges.json`** — entity-resolved co-affiliation network (2,663 nodes,
13,054 edges): one node per `Standardized_Institutions` value, with `degree`, `betweenness`,
`community` (greedy modularity), `in_science_park`, and geocoordinate fields. This is the primary
analysis network used throughout the paper's Results.

**`nodes.json` / `edges.json`** — naive co-affiliation network (10,404 nodes, 26,017 edges): one
node per raw affiliation string, unresolved. Used only as a methodological comparison (Table 1) to
demonstrate the effect of entity-resolution noise on the park-proximity association.

## Regenerating vs. reusing

Because Nominatim and the public Overpass API are volunteer-run, rate-limited services, we
recommend reusing the provided data files rather than re-running Steps 1–7 unless you are
extending the pipeline to a new OpenAlex topic or a larger sample.
