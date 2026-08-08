"""
15_reviewer_response_analyses.py - supplementary analyses for the reporting protocol

Runs nine analyses that extend the robustness suite in 10_robustness_checks.py:

  1. Hyper-authorship exclusion at four institution-count caps (>=6, >=8, >=10, >=15),
     each re-estimating the headline regression rather than the bivariate test only,
     with a full-model label-permutation p at 5,000 refits.
  2. Bootstrap confidence intervals and TOST equivalence tests for every
     decision-relevant coefficient, under both a model-based and a resampling-based
     standard error.
  3. Statistical power of the model-based test and of the label-permutation test on
     the consortium-excluded design, over a grid of pre-specified true effect sizes.
  4. Fractional counting: degree rebuilt so that a work listing k institutions
     contributes 1/(k-1) to each of its ties, with no work excluded.
  5. Country fixed effects.
  6. Betweenness pivot-seed stability.
  7. Rank-biserial effect size with a bootstrap confidence interval on both node
     schemes, so that the naive and entity-resolved effect sizes are compared on a
     scale that carries its own uncertainty rather than as two point estimates.
  8. Country-stratified label permutation, which reassigns the park label only
     within country groups, as a check on the exchangeability unit of the
     unrestricted permutation test.
  9. Affiliation multiplicity: the edge-level counterpart of node-identity error.
     A co-affiliation tie conflates two organizations collaborating with one author
     holding appointments at both, so the network is rebuilt keeping only ties that
     two distinct authors support.

Inputs (all produced by scripts 01-09 in this repository):
    affil_full.csv, park_matches.csv, combined.csv, geocoded.csv,
    std_nodes.json, density_5000m.csv, density_2000m_v2.csv

Outputs:
    reviewer_response_results.json  - every statistic reported below
    reviewer_response_tables.csv    - the same results in flat table form

The headline specification is a 5,000 m treatment radius with the university and
research-institute density covariate also measured at 5,000 m, both drawn from the
August 2026 OpenStreetMap retrieval, so that treatment and covariate share a scale
and a vintage. Negative binomial models are fitted with the dispersion parameter
estimated by maximum likelihood.

Run from the directory holding the inputs:
    python 15_reviewer_response_analyses.py
"""

import json
import math
import pickle
import time
import warnings
from collections import Counter, defaultdict
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats

warnings.filterwarnings("ignore")

SEARCH_RADIUS_M = 4000
COVARIATES = ["log_univ", "log_station", "log_junction"]
FORMULA = "degree ~ in_science_park + " + " + ".join(COVARIATES)

TREATMENT_RADIUS_M = 5000
EXCLUSION_CAPS = (6, 8, 10, 15)
POWER_GRID_IRR = (1.10, 1.15, 1.25, 1.50)
EQUIVALENCE_BOUND = (0.80, 1.25)

N_PERMUTATIONS = 5000
N_BOOTSTRAP = 2000
N_EFFECT_SIZE_BOOTSTRAP = 2000
N_POWER_SIMULATIONS = 3000
N_PERMUTATION_NULL = 3000
BETWEENNESS_PIVOTS = 500
BETWEENNESS_SEEDS = (42, 1, 7, 2026)

# Random-number seeds. These are arbitrary integers fixed for reproducibility.
# They are written in a date-like form only to keep them distinct and memorable;
# none of them encodes an execution date.
SEED_PERMUTATION = 20260808
SEED_BOOTSTRAP = 20260809
SEED_PERMUTATION_NULL = 20260811
SEED_POWER = 20260812
SEED_EFFECT_SIZE_BOOTSTRAP = 20260813
SEED_STRATIFIED_PERMUTATION = 20260814
SEED_MULTI_AFFILIATION = 20260815

MISCLASSIFIED_PARKS = "Australian Technology Park|Freiburg"


# --------------------------------------------------------------------------- #
# Data assembly
# --------------------------------------------------------------------------- #

def split_institutions(value):
    """Split a standardized-institution field on commas, as the pipeline does."""
    if pd.isna(value):
        return []
    return [s.strip() for s in str(value).split(",") if s.strip()]


def load_inputs():
    affil = pd.read_csv("affil_full.csv")
    park = pd.read_csv("park_matches.csv")
    combined = pd.read_csv("combined.csv")
    geocoded = pd.read_csv("geocoded.csv")
    nodes = json.load(open("std_nodes.json", encoding="utf-8"))
    density = pd.read_csv("density_5000m.csv")
    return affil, park, combined, geocoded, nodes, density


def density_by_affiliation(geocoded, density):
    """Map each raw affiliation string to its 5,000 m amenity count via coordinates."""
    by_coord = {
        (round(float(r.lat), 7), round(float(r.lon), 7)): float(r.univ_research_count_5000m)
        for r in density.itertuples()
    }
    out = {}
    for r in geocoded.itertuples():
        if pd.isna(r.Latitude) or pd.isna(r.Longitude):
            continue
        value = by_coord.get((round(float(r.Latitude), 7), round(float(r.Longitude), 7)))
        if value is not None:
            out[r.Raw_Affiliation] = value
    return out


def build_institution_lookup(affil, park, combined, radius_m, density_map):
    """Aggregate affiliation-level attributes to standardized institutions.

    Park status is assigned by majority vote across an institution's constituent
    affiliation strings; covariates take the first valid value, matching the
    procedure used throughout the pipeline.
    """
    rows = affil[["Paper_ID", "Raw_Affiliation", "Standardized_Institutions"]].dropna(
        subset=["Raw_Affiliation", "Standardized_Institutions"]
    )
    park_rows = park.set_index("Raw_Affiliation").to_dict(orient="index")
    covariate_rows = combined.set_index("Raw_Affiliation")[
        ["univ_research_count", "nearest_station_m", "nearest_junction_m"]
    ].to_dict(orient="index")

    verdict = {}
    for raw, row in park_rows.items():
        distance = row.get("distance_to_park_m")
        if distance is None or (isinstance(distance, float) and math.isnan(distance)):
            verdict[raw] = None
        else:
            verdict[raw] = (distance <= radius_m, row.get("nearest_park_name"))

    in_park = defaultdict(list)
    park_names = defaultdict(list)
    univ = defaultdict(list)
    station = defaultdict(list)
    junction = defaultdict(list)
    papers = defaultdict(set)

    for row in rows.itertuples():
        names = split_institutions(row.Standardized_Institutions)
        v = verdict.get(row.Raw_Affiliation)
        if v is not None:
            for name in names:
                in_park[name].append(v[0])
                if v[0]:
                    park_names[name].append(v[1])
        cov = covariate_rows.get(row.Raw_Affiliation)
        if cov is not None:
            count = density_map.get(row.Raw_Affiliation)
            for name in names:
                if count is not None and not (isinstance(count, float) and math.isnan(count)):
                    univ[name].append(count)
                for source, target in ((cov.get("nearest_station_m"), station),
                                       (cov.get("nearest_junction_m"), junction)):
                    if source is not None and not (isinstance(source, float) and math.isnan(source)):
                        target[name].append(source)
        for name in names:
            papers[name].add(row.Paper_ID)

    lookup = {}
    for name in set(in_park) | set(univ) | set(papers):
        votes = in_park.get(name, [])
        names_seen = park_names.get(name, [])
        lookup[name] = {
            "in_science_park": (sum(votes) > len(votes) / 2) if votes else False,
            "park_name": max(set(names_seen), key=names_seen.count) if names_seen else None,
            "univ_research_count": univ[name][0] if univ.get(name) else None,
            "nearest_station_m": station[name][0] if station.get(name) else None,
            "nearest_junction_m": junction[name][0] if junction.get(name) else None,
            "paper_count": len(papers.get(name, set())),
        }
    return lookup


def build_frame(nodes, lookup, degree=None):
    rows = []
    for node in nodes:
        attrs = lookup.get(node["id"])
        if attrs is None or attrs["univ_research_count"] is None:
            continue
        rows.append({
            "id": node["id"],
            "degree": degree.get(node["id"], 0) if degree is not None else node["degree"],
            "in_science_park": int(bool(attrs["in_science_park"])),
            "park_name": attrs["park_name"],
            "univ_research_count": attrs["univ_research_count"],
            "nearest_station_m": attrs["nearest_station_m"] if attrs["nearest_station_m"] is not None else SEARCH_RADIUS_M,
            "nearest_junction_m": attrs["nearest_junction_m"] if attrs["nearest_junction_m"] is not None else SEARCH_RADIUS_M,
            "paper_count": attrs["paper_count"],
        })
    frame = pd.DataFrame(rows)
    frame["log_univ"] = np.log1p(frame["univ_research_count"])
    frame["log_station"] = np.log1p(frame["nearest_station_m"])
    frame["log_junction"] = np.log1p(frame["nearest_junction_m"])
    frame["log_papers"] = np.log1p(frame["paper_count"])
    return frame


def papers_to_institutions(affil):
    mapping = defaultdict(set)
    for row in affil[["Paper_ID", "Standardized_Institutions"]].dropna().itertuples():
        for name in split_institutions(row.Standardized_Institutions):
            mapping[row.Paper_ID].add(name)
    return {k: sorted(v) for k, v in mapping.items()}


def rebuild_degree(paper_institutions, exclude_at=None, fractional=False):
    """Rebuild institutional degree, optionally excluding hyper-authored works or
    weighting each work's ties by 1/(k-1)."""
    if fractional:
        weighted = defaultdict(float)
        for institutions in paper_institutions.values():
            k = len(institutions)
            if k < 2 or (exclude_at is not None and k >= exclude_at):
                continue
            share = 1.0 / (k - 1)
            for a, b in combinations(institutions, 2):
                weighted[a] += share
                weighted[b] += share
        return dict(weighted)

    neighbours = defaultdict(set)
    for institutions in paper_institutions.values():
        k = len(institutions)
        if k < 2 or (exclude_at is not None and k >= exclude_at):
            continue
        for a, b in combinations(institutions, 2):
            neighbours[a].add(b)
            neighbours[b].add(a)
    return {k: len(v) for k, v in neighbours.items()}


def institutions_present(paper_institutions, exclude_at):
    present = set()
    for institutions in paper_institutions.values():
        if len(institutions) >= exclude_at:
            continue
        present.update(institutions)
    return present


# --------------------------------------------------------------------------- #
# Estimation
# --------------------------------------------------------------------------- #

def design_matrix(frame, covariates=COVARIATES):
    X = np.column_stack(
        [np.ones(len(frame)), frame["in_science_park"].values.astype(float)]
        + [frame[c].values for c in covariates]
    )
    return X, frame["degree"].values.astype(float)


def mle_dispersion(frame, covariates=COVARIATES):
    formula = "degree ~ in_science_park + " + " + ".join(covariates)
    response = frame["degree"]
    if not np.allclose(response, response.round()):
        frame = frame.assign(degree=response.round().astype(int))
    return smf.negativebinomial(formula, data=frame).fit(disp=0).params["alpha"]


def park_coefficient(X, y, alpha):
    model = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=alpha)).fit()
    return model.params[1], model.bse[1], model.pvalues[1]


def summarise(frame, covariates=COVARIATES, label=""):
    alpha = mle_dispersion(frame, covariates)
    X, y = design_matrix(frame, covariates)
    model = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=alpha)).fit()
    coef = model.params[1]
    ci95 = model.conf_int()[1]
    ci90 = model.conf_int(alpha=0.10)[1]
    return {
        "label": label,
        "n": len(frame),
        "treated": int(frame["in_science_park"].sum()),
        "irr": math.exp(coef),
        "ci95_low": math.exp(ci95[0]), "ci95_high": math.exp(ci95[1]),
        "ci90_low": math.exp(ci90[0]), "ci90_high": math.exp(ci90[1]),
        "model_p": float(model.pvalues[1]),
        "coef": float(coef), "se": float(model.bse[1]), "alpha": float(alpha),
    }


def permutation_p(frame, covariates=COVARIATES, n=N_PERMUTATIONS, seed=SEED_PERMUTATION):
    """Full-model label permutation: refit the complete specification after each
    random reassignment of the park label, holding the degree sequence fixed."""
    alpha = mle_dispersion(frame, covariates)
    X, y = design_matrix(frame, covariates)
    observed = abs(park_coefficient(X, y, alpha)[0])
    rng = np.random.default_rng(seed)
    labels = X[:, 1].copy()
    exceed = 0
    for _ in range(n):
        X[:, 1] = rng.permutation(labels)
        if abs(park_coefficient(X, y, alpha)[0]) >= observed:
            exceed += 1
    X[:, 1] = labels
    return {"permutation_p": (exceed + 1) / (n + 1), "exceedances": exceed, "refits": n, "seed": seed}


def bootstrap_interval(frame, covariates=COVARIATES, n=N_BOOTSTRAP, seed=SEED_BOOTSTRAP):
    """Nonparametric bootstrap over institutions."""
    alpha = mle_dispersion(frame, covariates)
    X, y = design_matrix(frame, covariates)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        idx = rng.integers(0, len(frame), len(frame))
        try:
            draws.append(park_coefficient(X[idx], y[idx], alpha)[0])
        except Exception:
            continue
    draws = np.array(draws)
    return {
        "replicates": len(draws),
        "ci90_low": float(np.exp(np.percentile(draws, 5))),
        "ci90_high": float(np.exp(np.percentile(draws, 95))),
        "ci95_low": float(np.exp(np.percentile(draws, 2.5))),
        "ci95_high": float(np.exp(np.percentile(draws, 97.5))),
        "se_log": float(draws.std(ddof=1)),
        "seed": seed,
    }


def tost(coef, se, bound=EQUIVALENCE_BOUND):
    """Two one-sided tests against an equivalence bound expressed as rate ratios."""
    low, high = math.log(bound[0]), math.log(bound[1])
    p_upper = float(stats.norm.cdf((coef - high) / se))
    p_lower = float(1 - stats.norm.cdf((coef - low) / se))
    p = max(p_upper, p_lower)
    return {"p_upper": p_upper, "p_lower": p_lower, "p_tost": p, "equivalent": p < 0.05}


def power_grid(frame, grid=POWER_GRID_IRR, covariates=COVARIATES,
               n_sim=N_POWER_SIMULATIONS, n_null=N_PERMUTATION_NULL):
    """Power of the model-based test and of the permutation test.

    The permutation critical value is taken from the empirical 95th percentile of
    the absolute park coefficient under label permutation, which is materially
    larger than a normal approximation would give because the permutation null is
    heavy-tailed under dyadic non-independence.
    """
    alpha = mle_dispersion(frame, covariates)
    X, _ = design_matrix(frame, covariates)
    y = frame["degree"].values.astype(float)

    rng_null = np.random.default_rng(SEED_PERMUTATION_NULL)
    labels = X[:, 1].copy()
    null_draws = []
    for _ in range(n_null):
        X[:, 1] = rng_null.permutation(labels)
        null_draws.append(park_coefficient(X, y, alpha)[0])
    X[:, 1] = labels
    null_draws = np.abs(np.array(null_draws))
    critical = float(np.percentile(null_draws, 95))

    null_formula = "degree ~ " + " + ".join(covariates)
    fitted = smf.glm(null_formula, data=frame,
                     family=sm.families.NegativeBinomial(alpha=alpha)).fit().fittedvalues.values
    treated = X[:, 1]
    rng = np.random.default_rng(SEED_POWER)
    size = 1.0 / alpha

    results = {}
    for irr in grid:
        mu = fitted * np.power(irr, treated)
        model_hits = permutation_hits = converged = 0
        for _ in range(n_sim):
            y_sim = rng.negative_binomial(size, size / (size + mu)).astype(float)
            try:
                coef, _, p = park_coefficient(X, y_sim, alpha)
            except Exception:
                continue
            converged += 1
            if p < 0.05 and coef > 0:
                model_hits += 1
            if coef >= critical:
                permutation_hits += 1
        results[irr] = {
            "power_model": model_hits / converged,
            "mc_se_model": math.sqrt((model_hits / converged) * (1 - model_hits / converged) / converged),
            "power_permutation": permutation_hits / converged,
            "mc_se_permutation": math.sqrt((permutation_hits / converged) * (1 - permutation_hits / converged) / converged),
            "replicates": converged,
        }
    return {
        "permutation_null_sd": float(null_draws.std(ddof=1)),
        "permutation_critical_value": critical,
        "critical_over_sd": critical / float(null_draws.std(ddof=1)),
        "grid": results,
    }


def build_resolved_network_frame(nodes):
    """Node-identity scheme 2: the entity-resolved network as 08_build_entity_resolved
    _network.py delivers it, with that script's own park assignment and degree.

    The effect-size comparison is a statement about the two networks as built, so
    both arms are read from their delivering pipeline rather than reclassified here.
    """
    return pd.DataFrame([{
        "id": node["id"],
        "degree": node["degree"],
        "in_science_park": int(bool(node["in_science_park"])),
    } for node in nodes])


def build_naive_frame(affil, park, combined):
    """Node-identity scheme 1: raw affiliation strings taken as institutions.

    This is the network of 09_build_naive_network.py. Every distinct affiliation
    string is its own node, so a single institution recorded under k spellings
    appears as k nodes, each carrying an independent copy of the same underlying
    tie structure. Park status is the merged Method A/B verdict recorded in
    park_matches.csv, and the covariates are the affiliation-level values in
    combined.csv, both taken as the naive pipeline takes them.
    """
    rows = affil[["Paper_ID", "Raw_Affiliation"]].dropna()
    by_paper = defaultdict(set)
    for row in rows.itertuples():
        by_paper[row.Paper_ID].add(row.Raw_Affiliation)

    neighbours = defaultdict(set)
    for institutions in by_paper.values():
        if len(institutions) < 2:
            continue
        for a, b in combinations(sorted(institutions), 2):
            neighbours[a].add(b)
            neighbours[b].add(a)
    degree = {k: len(v) for k, v in neighbours.items()}

    verdict = park.set_index("Raw_Affiliation")["in_park_best_guess"].astype(str).str.lower().eq("true").to_dict()
    covariates = combined.set_index("Raw_Affiliation")[
        ["univ_research_count", "nearest_station_m", "nearest_junction_m", "Latitude"]
    ].to_dict(orient="index")

    records = []
    for raw, cov in covariates.items():
        if _missing(cov["Latitude"]) or _missing(cov["univ_research_count"]):
            continue
        records.append({
            "id": raw,
            "degree": degree.get(raw, 0),
            "in_science_park": int(bool(verdict.get(raw, False))),
            "univ_research_count": cov["univ_research_count"],
            "nearest_station_m": SEARCH_RADIUS_M if _missing(cov["nearest_station_m"]) else cov["nearest_station_m"],
            "nearest_junction_m": SEARCH_RADIUS_M if _missing(cov["nearest_junction_m"]) else cov["nearest_junction_m"],
        })
    frame = pd.DataFrame(records)
    frame["log_univ"] = np.log1p(frame["univ_research_count"])
    frame["log_station"] = np.log1p(frame["nearest_station_m"])
    frame["log_junction"] = np.log1p(frame["nearest_junction_m"])
    return frame


def _missing(value):
    return value is None or (isinstance(value, float) and math.isnan(value))


def effect_size_bootstrap(frame, n=N_EFFECT_SIZE_BOOTSTRAP, seed=SEED_EFFECT_SIZE_BOOTSTRAP):
    """Rank-biserial correlation with a percentile bootstrap interval.

    The pseudo-replication claim is that node-identity error inflates certainty
    without moving the effect. Testing that claim requires an effect size carrying
    its own uncertainty, so the two arms are resampled independently and the
    rank-biserial statistic recomputed on each draw.
    """
    treated = frame.loc[frame["in_science_park"] == 1, "degree"].values
    control = frame.loc[frame["in_science_park"] == 0, "degree"].values
    u, p = stats.mannwhitneyu(treated, control, alternative="two-sided")
    observed = 2 * u / (len(treated) * len(control)) - 1

    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        a = rng.choice(treated, len(treated), replace=True)
        b = rng.choice(control, len(control), replace=True)
        uu, _ = stats.mannwhitneyu(a, b, alternative="two-sided")
        draws.append(2 * uu / (len(a) * len(b)) - 1)
    draws = np.array(draws)

    return {
        "n": len(frame),
        "treated": len(treated),
        "control": len(control),
        "rank_biserial": float(observed),
        "ci95_low": float(np.percentile(draws, 2.5)),
        "ci95_high": float(np.percentile(draws, 97.5)),
        "mann_whitney_p": float(p),
        "cles": float((observed + 1) / 2),
        "mean_degree_treated": float(treated.mean()),
        "mean_degree_control": float(control.mean()),
        "mean_degree_ratio": float(treated.mean() / control.mean()),
        "replicates": n,
        "seed": seed,
    }


def stratified_permutation_p(frame, countries, covariates=COVARIATES,
                             n=N_PERMUTATIONS, seed=SEED_STRATIFIED_PERMUTATION):
    """Country-stratified label permutation.

    Park status is spatially clustered, so an unrestricted permutation generates
    null draws with treatment geographies no registry could produce. Restricting
    the reassignment to within-country groups holds the between-country
    distribution of treatment fixed and changes the exchangeability unit from the
    institution to the institution-within-country.
    """
    frame = frame.copy()
    frame["country"] = frame["id"].map(countries).fillna("__none__")
    alpha = mle_dispersion(frame, covariates)
    X, y = design_matrix(frame, covariates)
    observed = abs(park_coefficient(X, y, alpha)[0])

    labels = X[:, 1].copy()
    groups = [np.where(frame["country"].values == g)[0] for g in frame["country"].unique()]
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(n):
        permuted = labels.copy()
        for index in groups:
            if len(index) > 1:
                permuted[index] = rng.permutation(labels[index])
        X[:, 1] = permuted
        if abs(park_coefficient(X, y, alpha)[0]) >= observed:
            exceed += 1
    X[:, 1] = labels

    return {
        "stratified_permutation_p": (exceed + 1) / (n + 1),
        "exceedances": exceed,
        "refits": n,
        "n": len(frame),
        "treated": int(frame["in_science_park"].sum()),
        "country_groups": int(frame.loc[frame["country"] != "__none__", "country"].nunique()),
        "institutions_without_country": int((frame["country"] == "__none__").sum()),
        "seed": seed,
    }


def author_supported_degree(affil):
    """Degree with dual-appointment-only ties removed.

    An institution pair on a work is retained only where two *distinct* authors
    support it. A pair linked solely because one author listed both institutions
    records a joint appointment, not a collaboration, and is dropped.
    """
    per_paper = defaultdict(lambda: defaultdict(set))
    rows = affil.dropna(subset=["Standardized_Institutions"])
    for row in rows.itertuples():
        for name in split_institutions(row.Standardized_Institutions):
            per_paper[row.Paper_ID][name].add(row.Author_ID)

    authorship_institutions = defaultdict(set)
    for row in rows.itertuples():
        for name in split_institutions(row.Standardized_Institutions):
            authorship_institutions[(row.Paper_ID, row.Author_ID)].add(name)

    all_ties = defaultdict(set)
    supported_ties = defaultdict(set)
    pairs = 0
    unsupported = 0
    for authors_by_institution in per_paper.values():
        names = sorted(authors_by_institution)
        if len(names) < 2:
            continue
        for a, b in combinations(names, 2):
            pairs += 1
            all_ties[a].add(b)
            all_ties[b].add(a)
            authors_a, authors_b = authors_by_institution[a], authors_by_institution[b]
            if len(authors_a) == 1 and len(authors_b) == 1 and authors_a == authors_b:
                unsupported += 1
            else:
                supported_ties[a].add(b)
                supported_ties[b].add(a)

    return {
        "degree_all": {k: len(v) for k, v in all_ties.items()},
        "degree_supported": {k: len(v) for k, v in supported_ties.items()},
        "authorships": len(authorship_institutions),
        "multi_affiliation_authorships": sum(
            1 for v in authorship_institutions.values() if len(v) > 1),
        "paper_institution_pairs": pairs,
        "dual_appointment_only_pairs": unsupported,
        "edges_all": sum(len(v) for v in all_ties.values()) // 2,
        "edges_author_supported": sum(len(v) for v in supported_ties.values()) // 2,
    }


def country_lookup(affil):
    counts = defaultdict(Counter)
    for row in affil[["Standardized_Institutions", "Institution_Countries"]].itertuples():
        institutions = split_institutions(row.Standardized_Institutions)
        countries = split_institutions(row.Institution_Countries)
        if not institutions:
            continue
        if len(countries) == len(institutions):
            for name, country in zip(institutions, countries):
                counts[name][country] += 1
        elif countries and len(set(countries)) == 1:
            for name in institutions:
                counts[name][countries[0]] += 1
    return {k: v.most_common(1)[0][0] for k, v in counts.items() if v}


def betweenness_seed_stability(paper_institutions, frame, seeds=BETWEENNESS_SEEDS,
                               pivots=BETWEENNESS_PIVOTS):
    """Betweenness is computed by pivot sampling on graphs above 3,000 nodes, so the
    reported estimate depends on the pivot seed. This quantifies that dependence."""
    weights = defaultdict(int)
    for institutions in paper_institutions.values():
        if len(institutions) < 2:
            continue
        for a, b in combinations(institutions, 2):
            weights[(a, b) if a < b else (b, a)] += 1
    graph = nx.Graph()
    for (a, b), w in weights.items():
        graph.add_edge(a, b, weight=w, cost=1.0 / w)

    out = {}
    for seed in seeds:
        centrality = nx.betweenness_centrality(graph, k=pivots, weight="cost", seed=seed)
        values = frame["id"].map(lambda i: centrality.get(i, 0.0))
        treated = values[frame["in_science_park"] == 1]
        control = values[frame["in_science_park"] == 0]
        u, p = stats.mannwhitneyu(treated, control, alternative="two-sided")
        out[seed] = {
            "mann_whitney_p": float(p),
            "rank_biserial": float(2 * u / (len(treated) * len(control)) - 1),
        }
    return {"nodes": graph.number_of_nodes(), "edges": graph.number_of_edges(),
            "pivots": pivots, "by_seed": out}


# --------------------------------------------------------------------------- #

def main():
    started = time.time()
    results = {"settings": {
        "treatment_radius_m": TREATMENT_RADIUS_M,
        "equivalence_bound_irr": list(EQUIVALENCE_BOUND),
        "permutation_refits": N_PERMUTATIONS,
        "bootstrap_replicates": N_BOOTSTRAP,
        "power_simulations": N_POWER_SIMULATIONS,
        "seeds": {"permutation": SEED_PERMUTATION, "bootstrap": SEED_BOOTSTRAP,
                  "permutation_null": SEED_PERMUTATION_NULL, "power": SEED_POWER,
                  "effect_size_bootstrap": SEED_EFFECT_SIZE_BOOTSTRAP,
                  "stratified_permutation": SEED_STRATIFIED_PERMUTATION,
                  "multi_affiliation_permutation": SEED_MULTI_AFFILIATION,
                  "betweenness_pivots": list(BETWEENNESS_SEEDS)},
        "seed_note": ("Seeds are arbitrary integers fixed for reproducibility. "
                      "The date-like form is a naming convention only and encodes "
                      "no execution date."),
    }}

    affil, park, combined, geocoded, nodes, density = load_inputs()
    density_map = density_by_affiliation(geocoded, density)
    lookup = build_institution_lookup(affil, park, combined, TREATMENT_RADIUS_M, density_map)
    headline = build_frame(nodes, lookup)
    paper_institutions = papers_to_institutions(affil)

    sizes = np.array([len(v) for v in paper_institutions.values()])
    results["corpus"] = {
        "papers_with_institutions": len(paper_institutions),
        "mean_institutions_per_paper": float(sizes.mean()),
        "median_institutions_per_paper": float(np.median(sizes)),
        "max_institutions_per_paper": int(sizes.max()),
        "share_at_or_above_cap": {c: float((sizes >= c).mean()) for c in EXCLUSION_CAPS},
    }

    results["censoring"] = {
        "affiliation_records": int(combined["Latitude"].notna().sum()),
        "station_at_bound_share": float(
            (combined.loc[combined["Latitude"].notna(), "nearest_station_m"].isna()).mean()),
        "junction_at_bound_share": float(
            (combined.loc[combined["Latitude"].notna(), "nearest_junction_m"].isna()).mean()),
        "station_at_bound_share_analysis_frame": float((headline["nearest_station_m"] >= SEARCH_RADIUS_M).mean()),
        "junction_at_bound_share_analysis_frame": float((headline["nearest_junction_m"] >= SEARCH_RADIUS_M).mean()),
    }

    # --- specifications ---------------------------------------------------- #
    specifications = {"headline": headline}
    for cap in EXCLUSION_CAPS:
        present = institutions_present(paper_institutions, cap)
        degree = rebuild_degree(paper_institutions, exclude_at=cap)
        frame = headline[headline["id"].isin(present)].copy()
        frame["degree"] = frame["id"].map(lambda i: degree.get(i, 0))
        specifications[f"cap_ge_{cap}"] = frame

    joint = specifications["cap_ge_10"].copy()
    bad = joint["park_name"].astype(str).str.contains(MISCLASSIFIED_PARKS, case=False, na=False)
    joint.loc[bad, "in_science_park"] = 0
    specifications["joint_perturbation"] = joint

    results["specifications"] = {}
    for label, frame in specifications.items():
        summary = summarise(frame, label=label)
        summary.update(permutation_p(frame))
        summary["bootstrap"] = bootstrap_interval(frame)
        summary["tost_model_se"] = tost(summary["coef"], summary["se"])
        summary["tost_bootstrap_se"] = tost(summary["coef"], summary["bootstrap"]["se_log"])
        summary["log_effect_removed"] = 1 - math.log(summary["irr"]) / math.log(
            results["specifications"]["headline"]["irr"]) if label != "headline" else 0.0
        results["specifications"][label] = summary
        print(f"{label:20s} n={summary['n']:5d} treated={summary['treated']:4d} "
              f"IRR={summary['irr']:.4f} model p={summary['model_p']:.4g} "
              f"permutation p={summary['permutation_p']:.4f} "
              f"TOST(model)={summary['tost_model_se']['p_tost']:.4f} "
              f"TOST(bootstrap)={summary['tost_bootstrap_se']['p_tost']:.4f}", flush=True)

    # --- productivity-controlled specifications ---------------------------- #
    with_productivity = COVARIATES + ["log_papers"]
    summary = summarise(headline, covariates=with_productivity, label="headline_with_productivity")
    summary.update(permutation_p(headline, covariates=with_productivity))
    summary["bootstrap"] = bootstrap_interval(headline, covariates=with_productivity)
    summary["tost_bootstrap_se"] = tost(summary["coef"], summary["bootstrap"]["se_log"])
    results["specifications"]["headline_with_productivity"] = summary
    print(f"headline+productivity IRR={summary['irr']:.4f} "
          f"permutation p={summary['permutation_p']:.4f}", flush=True)

    # --- fractional counting ----------------------------------------------- #
    fractional_degree = rebuild_degree(paper_institutions, fractional=True)
    fractional = headline.copy()
    fractional["degree"] = fractional["id"].map(lambda i: fractional_degree.get(i, 0.0))
    summary = summarise(fractional, label="fractional_counting")
    summary.update(permutation_p(fractional))
    summary["log_effect_removed"] = 1 - math.log(summary["irr"]) / math.log(
        results["specifications"]["headline"]["irr"])
    summary["mean_degree_whole"] = float(headline["degree"].mean())
    summary["mean_degree_fractional"] = float(fractional["degree"].mean())
    results["specifications"]["fractional_counting"] = summary
    print(f"fractional counting  IRR={summary['irr']:.4f} "
          f"permutation p={summary['permutation_p']:.4f}", flush=True)

    # --- country fixed effects --------------------------------------------- #
    countries = country_lookup(affil)
    with_country = headline.copy()
    with_country["country"] = with_country["id"].map(countries)
    dropped = int(with_country["country"].isna().sum())
    with_country = with_country[with_country["country"].notna()]
    alpha = mle_dispersion(with_country)
    model = smf.glm(FORMULA + " + C(country)", data=with_country,
                    family=sm.families.NegativeBinomial(alpha=alpha)).fit()
    coef = model.params["in_science_park"]
    ci = model.conf_int().loc["in_science_park"]
    results["country_fixed_effects"] = {
        "n": len(with_country), "treated": int(with_country["in_science_park"].sum()),
        "countries": int(with_country["country"].nunique()),
        "countries_with_treated": int((with_country.groupby("country")["in_science_park"].sum() > 0).sum()),
        "institutions_without_country": dropped,
        "irr": math.exp(coef), "ci95_low": math.exp(ci[0]), "ci95_high": math.exp(ci[1]),
        "model_p": float(model.pvalues["in_science_park"]),
    }
    print("country fixed effects", results["country_fixed_effects"], flush=True)

    # --- power and betweenness --------------------------------------------- #
    results["power"] = power_grid(specifications["cap_ge_10"])
    print("power", json.dumps(results["power"]["grid"], indent=1), flush=True)
    results["betweenness_seed_stability"] = betweenness_seed_stability(paper_institutions, headline)
    print("betweenness", results["betweenness_seed_stability"]["by_seed"], flush=True)

    # --- effect size on both node schemes ---------------------------------- #
    naive = build_naive_frame(affil, park, combined)
    resolved = build_resolved_network_frame(nodes)
    results["node_scheme_effect_size"] = {
        "naive": effect_size_bootstrap(naive),
        "entity_resolved": effect_size_bootstrap(resolved),
    }
    results["node_scheme_effect_size"]["collapse_ratio"] = (
        len(naive) / len(resolved))
    for label, s_ in results["node_scheme_effect_size"].items():
        print(f"{label:16s} rank-biserial r = {s_['rank_biserial']:.4f} "
              f"[{s_['ci95_low']:.4f}, {s_['ci95_high']:.4f}]  "
              f"Mann-Whitney p = {s_['mann_whitney_p']:.4g}  n = {s_['n']}", flush=True)

    # --- country-stratified permutation ------------------------------------ #
    results["stratified_permutation"] = {}
    for label in ("headline", "cap_ge_10"):
        out = stratified_permutation_p(specifications[label], countries)
        out["unrestricted_permutation_p"] = results["specifications"][label]["permutation_p"]
        results["stratified_permutation"][label] = out
        print(f"{label:16s} within-country permutation p = {out['stratified_permutation_p']:.4f} "
              f"against {out['unrestricted_permutation_p']:.4f} unrestricted "
              f"({out['country_groups']} country groups)", flush=True)

    # --- affiliation multiplicity ------------------------------------------ #
    multi = author_supported_degree(affil)
    supported = headline.copy()
    supported["degree"] = supported["id"].map(lambda i: multi["degree_supported"].get(i, 0))
    summary = summarise(supported, label="author_supported_ties")
    summary.update(permutation_p(supported, seed=SEED_MULTI_AFFILIATION))
    lost = headline["id"].map(
        lambda i: multi["degree_all"].get(i, 0) - multi["degree_supported"].get(i, 0))
    u, p_lost = stats.mannwhitneyu(lost[headline["in_science_park"] == 1],
                                   lost[headline["in_science_park"] == 0],
                                   alternative="two-sided")
    results["affiliation_multiplicity"] = {
        k: multi[k] for k in ("authorships", "multi_affiliation_authorships",
                              "paper_institution_pairs", "dual_appointment_only_pairs",
                              "edges_all", "edges_author_supported")}
    results["affiliation_multiplicity"].update({
        "specification": summary,
        "degrees_removed_treated": float(lost[headline["in_science_park"] == 1].mean()),
        "degrees_removed_control": float(lost[headline["in_science_park"] == 0].mean()),
        "degrees_removed_mann_whitney_p": float(p_lost),
    })
    print(f"affiliation multiplicity: {multi['multi_affiliation_authorships']} of "
          f"{multi['authorships']} authorships list >=2 institutions; "
          f"{multi['edges_all'] - multi['edges_author_supported']} of {multi['edges_all']} edges "
          f"dropped; IRR={summary['irr']:.4f} permutation p={summary['permutation_p']:.4f}",
          flush=True)

    json.dump(results, open("reviewer_response_results.json", "w"), indent=1, default=str)
    flat = []
    for label, s in results["specifications"].items():
        flat.append({
            "specification": label, "n": s["n"], "treated": s["treated"], "irr": s["irr"],
            "ci95_low": s["ci95_low"], "ci95_high": s["ci95_high"],
            "ci90_low": s["ci90_low"], "ci90_high": s["ci90_high"],
            "model_p": s["model_p"], "permutation_p": s["permutation_p"],
            "bootstrap_ci90_low": s.get("bootstrap", {}).get("ci90_low"),
            "bootstrap_ci90_high": s.get("bootstrap", {}).get("ci90_high"),
            "tost_p_model_se": s.get("tost_model_se", {}).get("p_tost"),
            "tost_p_bootstrap_se": s.get("tost_bootstrap_se", {}).get("p_tost"),
            "log_effect_removed": s.get("log_effect_removed"),
        })
    pd.DataFrame(flat).to_csv("reviewer_response_tables.csv", index=False)
    print(f"wrote reviewer_response_results.json and reviewer_response_tables.csv "
          f"({(time.time() - started) / 60:.1f} min)")


if __name__ == "__main__":
    main()
