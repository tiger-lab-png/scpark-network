# scpark-network

Reproducible pipeline for **"Pseudo-replication in affiliation-string co-authorship
networks: node identity manufactures certainty, hyperauthorship manufactures effect"**
(Hung-Chi Chang, Short Communication submitted to the *Journal of Informetrics*,
September 2026). The full fourteen-item reporting protocol and every robustness
check are given in the Supplementary Material (`docs/`).

The paper shows two ways an affiliation-string network manufactures a small effect.
Treating raw strings as nodes commits pseudo-replication in Hurlbert's (1984)
sense: the node set inflates 9.6-fold, the standardized effect is unchanged
(rank-biserial *r* = .114 vs .121) while the *p* value moves twenty orders of
magnitude, and fragmentation that differs between arms biases the estimate.
At the edge level, the 0.33% of works listing ten or more institutions carry
48.6% of the log-effect of a spatial covariate that had survived every check
conditioning on the units, and fractional counting removes 38.0% without
discarding a work. The science-park proximity question is the test-bed; the
fourteen-item reporting protocol from which the checks are drawn is Table S9 of
the Supplementary Material.

Everything reported in the manuscript can be re-executed from this repository.

---

## Quick start

```bash
pip install -r requirements.txt
python prepare_data.py     # decompress the deposited inputs (once)
python verify.py           # ~1 min: checks every deposited figure against the paper
```

`verify.py` exits non-zero if any deposited input is missing or if the
re-estimated headline specification does not match the manuscript. Add `--full`
to additionally re-run the 5,000-refit permutation analyses (1–2 hours).

---

## Which data belongs to which result

The manuscript reports **three** runs. They are deposited separately, and mixing
them will not reproduce anything.

| Directory | Contents | Where it appears in the paper |
|---|---|---|
| `data/full_run/` | **Full silicon-carbide population.** 54,671 works, 80,176 unique raw affiliation strings, 69,455 geocoded, 8,382 standardized institutions of which **7,886** carry valid coordinates. | **All inferential results.** Abstract, Table 1, Figures 1–2, Supplementary Material S1–S19. |
| `data/cart_run/` | CAR-T cell-therapy cross-field probe (capped sample). | Section 3.4 and Supplementary Material S15, S19 (Table S12). |
| `data/pilot_5000/` | The initial 5,000-work cursor-paginated sample: 10,404 raw strings, 2,663 standardized institutions (2,598 with coordinates). | Retained as the *designed sample-size comparison* described in Methodology, which separates effect-size stability from sample-size-driven changes in significance. It is **not** the analysis frame. |

Scripts read bare filenames, so run them from the directory holding the inputs:

```bash
cd data/full_run
python ../../analysis/15_reviewer_response_analyses.py
```

---

## Pipeline

Run in order. All intermediate outputs are deposited, so Steps 1–9 do not need to
be re-executed to reproduce any reported statistic.

| Step | Script | Purpose |
|---|---|---|
| 1 | `pipeline/01_fetch_openalex.py` | Cursor-paginated extraction of OpenAlex Topic **T10361** (Silicon Carbide Semiconductor Technologies), 2018–2024 → `affil_full.csv`, `addr_uniq_full.csv` |
| 1b | `pipeline/01b_fetch_openalex_cart.py` | Same extraction for the CAR-T cross-field probe |
| 2 | `pipeline/02_geocode_and_enrich.py` | Nominatim geocoding with the address-simplification cascade; Overpass `is_in()` park tagging and infrastructure-density covariates |
| 2b | `pipeline/02b_rerun_enrich_only.py` | Re-run the Overpass step without re-geocoding |
| 3 | `pipeline/03_fetch_wikidata_parks.py` | Wikidata SPARQL retrieval of 231 candidate science/technology parks |
| 4 | `pipeline/04_fetch_park_polygons.py` | Exact OSM polygon boundaries for the parks with a linked OSM relation |
| 5 | `pipeline/05_match_parks_distance.py` | Haversine distance classification (Method A) |
| 6 | `pipeline/06_apply_polygon_refinement.py` | Point-in-polygon refinement under the asymmetric confidence rule |
| 7 | `pipeline/07_merge_method_a_b.py` | Cross-validate Method A against Method B (OSM keyword tagging) |
| 8 | `pipeline/08_build_entity_resolved_network.py` | Entity-resolved network → `std_nodes.json`, `std_edges.json` (**primary analysis network**) |
| 9 | `pipeline/09_build_naive_network.py` | Naive raw-string network → `nodes.json`, `edges.json` (methodological comparison only) |

## Analysis

| Script | Reproduces |
|---|---|
| `analysis/10_robustness_checks.py` | Mann–Whitney tests, negative binomial regression, TOST equivalence, VIF, distance-threshold sweep, leave-one-park-out, label permutation, community composition, geocoding failure by country, Holm–Bonferroni |
| `analysis/11_institution_size_control.py` | Institution-productivity-controlled regression with MLE dispersion and 5-predictor VIF |
| `analysis/12_productivity_control_robustness.py` | Productivity-control robustness |
| `analysis/13_fetch_density_5000m.py` | 5,000 m density covariate retrieval (August 2026 OSM vintage) |
| `analysis/14_refetch_density_2000m.py` | Vintage-matched 2,000 m re-retrieval — establishes that the apparent radius effect was retrieval vintage, not spatial scale |
| `analysis/15_reviewer_response_analyses.py` | Hyper-authorship exclusion caps, bootstrap CIs and TOST, power grid, fractional counting, country fixed effects, betweenness seed stability, rank-biserial effect sizes, country-stratified permutation, affiliation-multiplicity audit |

### Manuscript location → script

| Manuscript location | Run from | Script |
|---|---|---|
| Table 1 — node-identity comparison, effect sizes (Section 3.1) | `data/full_run/` | `analysis/15_reviewer_response_analyses.py` |
| Figure 2 and Supplementary Table S1 — institution-count caps, fractional counting, joint perturbation, 2,000 m rows, placebo row (Section 3.3) | `data/full_run/` | `analysis/15_reviewer_response_analyses.py`, `analysis/16_review_response_2026_09.py`; plotted by `figures/fig2_forest_caps.py` |
| Supplementary S20–S22 and Figure S1 — placebo exclusion, covariate attenuation, geography of hyperauthored works, 5,000-work subsample benchmark, permutation-null diagnostics, catchment clustering, sandwich SEs; CAR-T cap and fractional rows of Table S12 | `data/full_run/`, `data/cart_run/` | `analysis/16_review_response_2026_09.py`, `analysis/17_review_response_round2.py` (outputs in `analysis/review_response_2026_09/`) |
| Section 3.2 and Supplementary Tables S10–S11 — baseline regression, threshold sweep, productivity control | `data/full_run/` | `analysis/10_robustness_checks.py`, `analysis/12_productivity_control_robustness.py` |
| Section 3.4 and Supplementary Table S12 — CAR-T cross-field probe | `data/cart_run/` | `analysis/10_robustness_checks.py`, `analysis/15_reviewer_response_analyses.py` |
| Exclusion caps, equivalence tests, power grid, fractional counting, country fixed effects, betweenness seed stability, stratified permutation, affiliation-multiplicity audit | `data/full_run/` | `analysis/15_reviewer_response_analyses.py` |
| Supplementary Material, Tables S1–S8 | `data/full_run/` | `analysis/15_reviewer_response_analyses.py` |
| Figure 1 — degree-centrality distributions under one labelling rule | `data/full_run/` | `figures/fig1_degree_distributions.py` (the older `fig3_degree_distributions.py` and its `fig3_data.json` are retained for reference) |
| Analysis-pipeline diagram (not in the Short Communication) | — | `figures/fig1_analysis_pipeline.dot` (`dot -Tpng -Gdpi=300`) |

All random seeds are module constants at the top of
`analysis/15_reviewer_response_analyses.py` and are listed in the Supplementary
Material, Table S8 (S14). They are arbitrary integers fixed for reproducibility; the eight-digit
form is a naming convention and encodes no execution date.

---

## Environment

```bash
pip install -r requirements.txt
```

Developed on Python 3.13.3 with the versions pinned in `requirements.txt`
(numpy 2.2.6, pandas 2.2.3, SciPy 1.15.2, statsmodels 0.14.6, NetworkX 3.4.2).
Reported estimates were independently reproduced under Python 3.11.15 with
numpy 2.3.5, pandas 3.0.2, SciPy 1.17.1 and NetworkX 3.6.1, so they are not
sensitive to numerical- or graph-library versions.

Files larger than 3 MB are stored gzipped to stay well below the 100 MB
per-file hosting limit; `prepare_data.py` restores them in place.

---

## Known limitations and audit notes

These are stated at length in the manuscript; the operative ones for anyone
re-using this code are:

- `Standardized_Institutions` is OpenAlex's own automated entity-resolution
  output, not a manually verified ground truth. Residual fragmentation was
  audited against ROR identifiers on a 5,000-work pilot extraction
  (Supplementary Material S1); the full-population, edge-network audit of
  residual fragmentation is reported in S13.
- Geocoding succeeds for 86.6% of the 80,176 raw affiliation strings in the full
  population. Failure is not geographically random; rates are tabulated by
  institution country in the Results.
- Wikidata park coverage (231 registry entries) is the design's weakest
  component, and OSM polygon boundaries exist for only a minority of them.
  The distance proxy's internally measured false-positive rate is 41.5% at
  2,000 m and 54.1% at 5,000 m against exact geometry.
- Nominatim and the public Overpass API are volunteer-run and rate-limited.
  Re-running Steps 1–7 against live services takes several hours and is not
  required to reproduce any reported statistic.

## Supplementary material

`docs/Supplementary_Material.docx` is the supplement accompanying the Short
Communication (Sections S0–S22, Tables S0–S12, Figure S1); Table S0 indexes it
by main-text section.

## Licence and citation

MIT (see `LICENSE`). Cite via `CITATION.cff` or:

> Chang, H.-C. (2026). *scpark-network: Analysis pipeline for science-park
> proximity and academic co-authorship networks* [Computer software]. Zenodo.
