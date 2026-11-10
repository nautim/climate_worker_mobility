"""
Shared utilities for the climate_worker_mobility notebook pipeline.

Usage in each notebook (add to the imports cell):
    from utils import load_counties, load_cbsa, set_plot_style, ...
"""

import gc
import glob
import itertools
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import libpysal
import esda
from scipy.stats import norm


# ──────────────────────────────────────────────────────────
# Geographic data loaders
# ──────────────────────────────────────────────────────────

def load_counties(data_loaded: str = "../data_loaded", crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """
    Load and reconcile the 2018 and 2024 US county shapefiles.

    Handles the Valdez-Chena Borough split: removes the 2018 Valdez
    polygon (which disappears in 2024) and adds the two successor
    boroughs (Chugach, Copper River) from the 2024 file.

    Parameters
    ----------
    data_loaded : str
        Relative path to the data_loaded directory (default works from notebooks_main/).
    crs : str
        Output CRS.

    Returns
    -------
    GeoDataFrame with STATEFP, COUNTYFP, NAME, STCOFIPS (5-char zero-padded string).
    """
    c18 = gpd.read_file(f"{data_loaded}/county/cb_2018_us_county_500k.shp")
    c24 = gpd.read_file(f"{data_loaded}/county/cb_2024_us_county_500k.shp")

    mask_valdez = (
        (c18["STATEFP"] == "02") &
        (c18["NAME"].str.contains("Valdez", case=False))
    )
    c18_clean = c18.loc[~mask_valdez].copy()

    ak24 = c24[c24["STATEFP"] == "02"].copy()
    add_rows = ak24[ak24["NAME"].isin(["Chugach", "Copper River"])]

    counties = pd.concat([c18_clean, add_rows], ignore_index=True)
    counties = counties.set_crs(c18.crs, allow_override=True).to_crs(crs)

    counties["STCOFIPS"] = (
        counties["STATEFP"].astype(str).str.zfill(2)
        + counties["COUNTYFP"].astype(str).str.zfill(3)
    )

    state_to_fips = {
        "AL": "01","AK": "02","AZ": "04","AR": "05","CA": "06","CO": "08","CT": "09",
        "DE": "10","DC": "11","FL": "12","GA": "13","HI": "15","ID": "16","IL": "17",
        "IN": "18","IA": "19","KS": "20","KY": "21","LA": "22","ME": "23","MD": "24",
        "MA": "25","MI": "26","MN": "27","MS": "28","MO": "29","MT": "30","NE": "31",
        "NV": "32","NH": "33","NJ": "34","NM": "35","NY": "36","NC": "37","ND": "38",
        "OH": "39","OK": "40","OR": "41","PA": "42","RI": "44","SC": "45","SD": "46",
        "TN": "47","TX": "48","UT": "49","VT": "50","VA": "51","WA": "53","WV": "54",
        "WI": "55","WY": "56","PR": "72"
    }
    fips_to_state = {v: k for k, v in state_to_fips.items()}
    
    counties["STATE"] = counties["STATEFP"].map(fips_to_state)
    
    counties["name"] = counties["NAME"] + ", " + counties["STATE"]

    del c18, c24
    gc.collect()
    return counties


def load_cbsa(path: str = "../data_loaded/cbsa_2023.xlsx") -> pd.DataFrame:
    """
    Load the OMB CBSA delineation Excel file and produce clean FIPS codes.

    Parameters
    ----------
    path : str
        Path to cbsa_2023.xlsx (default works from notebooks_main/).

    Returns
    -------
    DataFrame with 'CBSA Code', 'CBSA Title', 'FIPS State Code',
    'FIPS County Code', 'STCOFIPS' (5-char zero-padded string).
    """
    cbsa = pd.read_excel(path, header=1)
    cbsa = cbsa.dropna(subset=["FIPS County Code"])

    cbsa["FIPS State Code"] = cbsa["FIPS State Code"].apply(
        lambda x: str(int(x)) if pd.notna(x) else None
    )
    cbsa["FIPS County Code"] = cbsa["FIPS County Code"].apply(
        lambda x: str(int(x)) if pd.notna(x) else None
    )
    cbsa["STCOFIPS"] = (
        cbsa["FIPS State Code"].astype(str).str.zfill(2)
        + cbsa["FIPS County Code"].astype(str).str.zfill(3)
    )
    return cbsa


def find_nri_future_risk(directory: str = "../data_loaded/NRI_FutureRisk") -> str:
    """
    Return the path to the NRI Future Risk Excel file by glob, avoiding
    the hard-coded date embedded in the filename.

    Parameters
    ----------
    directory : str

    Returns
    -------
    str – path to the first matching file.

    Raises
    ------
    FileNotFoundError if no matching file is found.
    """
    pattern = f"{directory}/NRI_Future_Risk_Master_Datasheet_*.xlsx"
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No NRI Future Risk file found matching: {pattern}")
    if len(matches) > 1:
        warnings.warn(f"Multiple NRI Future Risk files found; using: {matches[0]}")
    return matches[0]


# ──────────────────────────────────────────────────────────
# Lookup dictionaries
# ──────────────────────────────────────────────────────────

NRI_HAZARD_NAMES: dict = {
    "CWAV_RISKS": "Cold Wave",
    "DRGT_RISKS": "Drought",
    "HWAV_RISKS": "Heat Wave",
    "VLCN_RISKS": "Volcano",
    "HAIL_RISKS": "Hail",
    "WNTW_RISKS": "Winter Weather",
    "ISTM_RISKS": "Ice Storm",
    "SWND_RISKS": "Strong Wind",
    "TRND_RISKS": "Tornado",
    "AVLN_RISKS": "Avalanche",
    "CFLD_RISKS": "Coastal Flooding",
    "HRCN_RISKS": "Hurricane",
    "TSUN_RISKS": "Tsunami",
    "LTNG_RISKS": "Lightning",
    "RISK_SCORE": "Overall Risk Score",
    "RFLD_RISKS": "Riverine Flooding",
    "WFIR_RISKS": "Wildfire",
    "ERQK_RISKS": "Earthquake",
    "LNDS_RISKS": "Landslide",
}
"""19-entry dict mapping NRI column codes to human-readable hazard names."""


SOC2_MAP: dict = {
    "11": "Management",
    "15": "Computer & Math",
    "13": "Business & Finance",
    "27": "Arts / Media / Design",
    "25": "Education / Training",
    "41": "Sales",
    "No": "No ONET code",
    "23": "Legal",
    "17": "Architecture & Engineering",
    "29": "Healthcare Practitioners",
    "43": "Office & Admin Support",
    "19": "Sciences",
    "21": "Social Services",
    "53": "Transportation",
    "31": "Healthcare Support",
    "39": "Personal Care",
    "49": "Installation & Repair",
    "51": "Production",
    "33": "Protective Services",
    "35": "Food Service",
    "47": "Construction",
    "37": "Cleaning",
    "45": "Farming",
}
"""23-entry dict mapping 2-digit SOC prefix to broad occupational category."""


# ──────────────────────────────────────────────────────────
# Plotting helpers
# ──────────────────────────────────────────────────────────

def set_plot_style() -> None:
    """
    Apply the project-standard matplotlib / seaborn style.

    Call once near the top of each notebook, after imports.
    """
    sns.set_theme(context="paper", style="white")
    plt.rcParams.update({
        # Font
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],

        # Axes
        "axes.edgecolor": "black",
        "axes.linewidth": 1.2,
        "axes.titlesize": 12,
        "axes.labelsize": 11,

        # Ticks
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,

        # Legend
        "legend.fontsize": 10,
        "legend.frameon": False,

        # Figure
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.transparent": False,

        # Remove gridlines
        "axes.grid": False,
        "grid.linewidth": 0.0,
    })


def make_stars(p: float) -> str:
    """
    Return APA-style significance stars for a p-value.

    *** p<0.001 | ** p<0.01 | * p<0.05 | + p<0.1 | '' otherwise.
    """
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    elif p < 0.1:
        return "+"
    return ""


# ──────────────────────────────────────────────────────────
# Analysis helpers
# ──────────────────────────────────────────────────────────

def classify_per_window(df: pd.DataFrame, centrality: str = "pagerank",
                        log_fn=np.log) -> pd.DataFrame:
    """
    Classify nodes into Brain / Mass / Dual / Peripheral hubs.

    Uses the diagonal log-space residual (d) and mean log-centrality (m)
    to assign quadrant labels based on quantile thresholds.

    Parameters
    ----------
    df : DataFrame
        Must contain columns f"{centrality}_exp" and f"{centrality}_irs".
    centrality : str
        Base name of the centrality columns, e.g. "pagerank".
    log_fn : callable
        Logarithm function to use. Default is np.log (natural log).
        Pass np.log10 to replicate the base-10 variant used in figures_final.

    Returns
    -------
    Copy of df with new columns 'd', 'm', and f"class_diag_{centrality}".
    """
    df = df.copy()
    exp_col = f"{centrality}_exp"
    irs_col = f"{centrality}_irs"
    EPS = 1e-9

    df["d"] = log_fn(df[exp_col] + EPS) - log_fn(df[irs_col] + EPS)
    df["m"] = 0.5 * (log_fn(df[exp_col] + EPS) + log_fn(df[irs_col] + EPS))

    df[f"class_diag_{centrality}"] = np.select(
        [
            df["d"] > df["d"].quantile(0.95),
            df["d"] < df["d"].quantile(0.1),
            df["m"] > df["m"].quantile(0.95),
        ],
        ["Brain hub", "Mass hub", "Dual hub"],
        default="Peripheral",
    )
    return df


def compute_ccdf(values) -> tuple:
    """
    Compute the complementary CDF P(X >= x).

    Accepts either a dict (values taken from dict.values()) or an array-like.
    Non-positive values are dropped. Returns arrays sorted ascending.

    Parameters
    ----------
    values : array-like or dict

    Returns
    -------
    (sorted_vals, ccdf_vals) – sorted_vals ascending, ccdf_vals = P(X >= x).
    """
    if isinstance(values, dict):
        arr = np.array(list(values.values()), dtype=float)
    else:
        arr = np.asarray(values, dtype=float)

    arr = arr[arr > 0]
    arr_sorted = np.sort(arr)
    n = len(arr_sorted)
    ccdf = 1.0 - np.arange(1, n + 1) / n
    return arr_sorted, ccdf


def build_full_dyad_grid(nodes, o_name: str = "origin", d_name: str = "destination") -> pd.DataFrame:
    """
    Build the full ordered origin x destination grid over `nodes`, excluding
    self-pairs (i == j).

    Used to construct a zero-inclusive flow panel (e.g. for a PPML gravity
    model) from a sparse edge list that only contains realized flows.

    Parameters
    ----------
    nodes : array-like
        Node identifiers (e.g. CBSA or county codes).
    o_name, d_name : str
        Column names for the origin and destination sides of the grid.

    Returns
    -------
    DataFrame with columns [o_name, d_name], one row per ordered pair i != j.
    """
    nodes = pd.unique(np.asarray(nodes))
    pairs = [(o, d) for o, d in itertools.product(nodes, nodes) if o != d]
    return pd.DataFrame(pairs, columns=[o_name, d_name])


_BILATERAL_CLUSTER_VCOV = {"CRV1": "origin_fe+destination_fe"}


def _build_zero_inclusive_grid(edges: pd.DataFrame, o_col: str, d_col: str,
                               coords: pd.DataFrame, nodes=None) -> pd.DataFrame:
    """
    Build the dense origin x destination dyad grid underlying a PPML fit:
    every ordered pair over `nodes` (default: the nodes actually appearing
    in `edges`), with observed flow `n` left-joined and missing flow filled
    with 0, plus centroid distance.

    Passing an explicit `nodes` list (e.g. the intersection of two networks'
    node sets) restricts the grid to that shared population, so results
    across networks are not confounded by differing node coverage.
    """
    if nodes is None:
        nodes = pd.unique(edges[[o_col, d_col]].values.ravel())
    grid = build_full_dyad_grid(nodes, o_col, d_col)
    grid = grid.merge(edges[[o_col, d_col, "n"]], on=[o_col, d_col], how="left")
    grid["n"] = grid["n"].fillna(0)
    grid = grid.merge(coords, left_on=o_col, right_index=True).rename(columns={"x": "x1", "y": "y1"})
    grid = grid.merge(coords, left_on=d_col, right_index=True).rename(columns={"x": "x2", "y": "y2"})
    grid["dist"] = np.hypot(grid["x2"] - grid["x1"], grid["y2"] - grid["y1"])
    grid["log_dist"] = np.log(grid["dist"])
    return grid.rename(columns={o_col: "origin_fe", d_col: "destination_fe"})


def _fit_ppml_pair(grid: pd.DataFrame) -> tuple:
    """
    Fit PPML without and with origin+destination fixed effects on `grid`,
    both with bilateral (origin+destination) clustered standard errors
    (Cameron, Gelbach & Miller two-way clustering, via pyfixest CRV1).
    """
    import pyfixest as pf

    m_no_fe = pf.fepois("n ~ log_dist", data=grid, vcov=_BILATERAL_CLUSTER_VCOV)
    m_fe = pf.fepois("n ~ log_dist | origin_fe + destination_fe", data=grid,
                     vcov=_BILATERAL_CLUSTER_VCOV)
    return m_no_fe, m_fe


def _ppml_row(spec: str, model, varname: str = "log_dist") -> dict:
    ci = model.confint().loc[varname]
    return {"spec": spec, "beta_log_dist": model.coef()[varname], "se": model.se()[varname],
            "p": model.pvalue()[varname], "ci_low": ci.iloc[0], "ci_high": ci.iloc[1], "n": model._N}


def _ols_row(spec: str, ols_res) -> dict:
    ci = ols_res.conf_int().loc["dist"]
    return {"spec": spec, "beta_log_dist": ols_res.params["dist"], "se": ols_res.bse["dist"],
            "p": ols_res.pvalues["dist"], "ci_low": ci[0], "ci_high": ci[1], "n": int(ols_res.nobs)}


def ppml_distance_decay_check(edges: pd.DataFrame, o_col: str, d_col: str,
                              coords: pd.DataFrame, ols_res, label: str = "") -> tuple:
    """
    Robustness check for a log-log distance-decay spec (SI): compares the
    existing OLS on log(flow) ~ log(dist) - which drops every zero-flow
    dyad by construction, since log(0) is undefined - against PPML fit on
    the full zero-inclusive dyad panel, with and without origin/destination
    fixed effects (Silva & Tenreyro, 2006). PPML standard errors are
    bilaterally clustered by origin and destination.

    Parameters
    ----------
    edges : DataFrame
        Sparse nonzero-flow edge list with columns [o_col, d_col, "n"].
    o_col, d_col : str
        Origin / destination id column names in `edges`.
    coords : DataFrame
        Indexed by node id, with columns ["x", "y"] (projected centroids).
    ols_res : statsmodels RegressionResults
        Fitted result of sm.OLS(log(n), const + log(dist)) on `edges`.
    label : str
        Descriptive label for the printed report (e.g. "High-skill (exp)").

    Returns
    -------
    (comparison, grid) -- comparison is a DataFrame with one row per spec
    (OLS / PPML no-FE / PPML with FE); grid is the dense zero-inclusive
    dyad panel used to fit the PPML specs.
    """
    grid = _build_zero_inclusive_grid(edges, o_col, d_col, coords)
    n_possible = grid.shape[0]
    n_observed = int((grid["n"] > 0).sum())
    pct_zero = 100 * (1 - n_observed / n_possible)

    m_no_fe, m_fe = _fit_ppml_pair(grid)

    comparison = pd.DataFrame([
        _ols_row("(A) OLS log-log, zeros dropped [existing spec]", ols_res),
        _ppml_row("(B) PPML, zeros retained, no FE, bilateral-clustered SE", m_no_fe),
        _ppml_row("(C) PPML, zeros retained, O+D FE, bilateral-clustered SE [reviewer-requested spec]", m_fe),
    ])

    print(f"{label}: {n_possible:,} possible ordered pairs; {n_observed:,} have nonzero flow "
          f"-> {pct_zero:.1f}% of dyads are zero-flow and excluded from spec (A) by construction "
          f"of log(F).")
    print(comparison.to_string(index=False))

    return comparison, grid


def ppml_distance_decay_common_nodes(edges_a: pd.DataFrame, edges_b: pd.DataFrame,
                                     o_col: str, d_col: str, coords: pd.DataFrame,
                                     label_a: str = "", label_b: str = "") -> pd.DataFrame:
    """
    Re-run the OLS-vs-PPML distance-decay comparison for two networks (e.g.
    high-skill vs. general mobility) restricted to their shared node set, so
    the beta contrast between networks is not confounded by differing node
    coverage. For each network, both the OLS "bridge" model and the PPML
    models (no FE / O+D FE, bilateral-clustered SEs) are re-estimated on the
    identical, common-node dyad grid.

    Parameters
    ----------
    edges_a, edges_b : DataFrame
        Sparse nonzero-flow edge lists (e.g. high-skill, general mobility),
        each with columns [o_col, d_col, "n"].
    o_col, d_col : str
        Origin / destination id column names.
    coords : DataFrame
        Indexed by node id, with columns ["x", "y"] (projected centroids).
    label_a, label_b : str
        Descriptive labels for the two networks.

    Returns
    -------
    DataFrame with one row per (network, spec) combination.
    """
    nodes_a = set(pd.unique(edges_a[[o_col, d_col]].values.ravel()))
    nodes_b = set(pd.unique(edges_b[[o_col, d_col]].values.ravel()))
    common_nodes = np.array(sorted(nodes_a & nodes_b))

    print(f"Shared node set: {len(common_nodes)} nodes "
          f"({len(nodes_a)} in {label_a or 'network A'}, {len(nodes_b)} in {label_b or 'network B'}).")

    rows = []
    for label, edges in [(label_a, edges_a), (label_b, edges_b)]:
        edges_common = edges[edges[o_col].isin(common_nodes) & edges[d_col].isin(common_nodes)].copy()

        ols_res = sm.OLS(np.log(edges_common["n"]),
                         sm.add_constant(np.log(edges_common["dist"]))).fit()

        grid = _build_zero_inclusive_grid(edges_common, o_col, d_col, coords, nodes=common_nodes)
        n_possible = grid.shape[0]
        n_observed = int((grid["n"] > 0).sum())
        pct_zero = 100 * (1 - n_observed / n_possible)

        m_no_fe, m_fe = _fit_ppml_pair(grid)

        for row in [
            _ols_row("(A) OLS log-log, zeros dropped, common-node bridge", ols_res),
            _ppml_row("(B) PPML, zeros retained, no FE, bilateral-clustered SE", m_no_fe),
            _ppml_row("(C) PPML, zeros retained, O+D FE, bilateral-clustered SE", m_fe),
        ]:
            row["network"] = label
            row["pct_zero"] = pct_zero
            rows.append(row)

    comparison = pd.DataFrame(rows)[["network", "spec", "beta_log_dist", "se", "p",
                                     "ci_low", "ci_high", "n", "pct_zero"]]
    print(comparison.to_string(index=False))
    return comparison


def pagerank_gravity_factorisation_check(grid: pd.DataFrame, hits: pd.DataFrame,
                                         id_col: str = "CBSA", alpha: float = 0.85,
                                         top_k_exclude: int = 50, label: str = "") -> dict:
    """
    Tests whether real PageRank scores are well approximated by the gravity
    model's multiplicative decomposition PR_j ~ exp(nu_j) * accessibility_j,
    where nu_j is the destination fixed effect from the PPML gravity fit
    (log F_ij ~ log(dist_ij) | origin_fe + destination_fe, as in
    ppml_distance_decay_check / _fit_ppml_pair above), and accessibility_j is
    recovered implicitly by running PageRank on the gravity model's own
    fitted (response-scale) flow matrix.

    Reports two statistics (for SI, "Distance decay modelling" -
    "PageRank as an implied gravity destination effect"):
      (1) R^2 of log(real PageRank) regressed on the destination fixed
          effect nu_j alone - how much of PageRank is explained by gravity
          "attractiveness" net of distance and competing destinations;
      (2) R^2 and Spearman rho between log(real PageRank) and log(PageRank
          computed on the gravity model's fitted flow matrix) - how well the
          full gravity + network-recursion structure reproduces real
          PageRank, both on the full node set and after excluding the
          `top_k_exclude` highest-PageRank nodes (checking the fit isn't
          driven by a handful of hubs).

    Parameters
    ----------
    grid : DataFrame
        The dense zero-inclusive dyad grid already built for the PPML fit
        (the second element returned by ppml_distance_decay_check), with
        columns ["origin_fe", "destination_fe", "log_dist", "n"].
    hits : DataFrame
        PageRank output (e.g. cbsa2_hits_exp_1122.parquet / cbsa2_hits_irs_1122.parquet),
        with an id column (default "CBSA") and a "pagerank" column.
    id_col : str
        Name of the node-id column in `hits`.
    alpha : float
        PageRank damping factor; must match the value used to compute `hits`
        (0.85 throughout this project).
    top_k_exclude : int
        Number of highest-PageRank nodes to additionally exclude, as a
        robustness check.
    label : str

    Returns
    -------
    dict with beta_log_dist, n_destination_fe, r2_destination_fe,
    n_gravity_pagerank, r2_gravity_pagerank, spearman_gravity_pagerank,
    and r2_gravity_pagerank_excl_topK.
    """
    import pyfixest as pf
    import networkx as nx
    from scipy.stats import spearmanr

    m_fe = pf.fepois("n ~ log_dist | origin_fe + destination_fe", data=grid,
                     vcov=_BILATERAL_CLUSTER_VCOV)
    beta = m_fe.coef()["log_dist"]

    fe_dict = m_fe.fixef()
    dest_key = [k for k in fe_dict if "destination_fe" in k][0]
    nu = pd.Series(fe_dict[dest_key])
    nu.index = nu.index.astype(str)

    real_pr = hits.set_index(id_col)["pagerank"].copy()
    real_pr.index = real_pr.index.astype(str)

    common_nu = nu.index.intersection(real_pr.index)
    log_pr_nu = np.log(real_pr.loc[common_nu])
    ols_nu = sm.OLS(log_pr_nu.values, sm.add_constant(nu.loc[common_nu].values)).fit()
    r2_nu = ols_nu.rsquared

    grid = grid.copy()
    grid["n_hat"] = m_fe.predict(grid, type="response")
    G = nx.DiGraph()
    for o, d, w in zip(grid["origin_fe"], grid["destination_fe"], grid["n_hat"]):
        if w > 0:
            G.add_edge(o, d, weight=float(w))
    pr_gravity = pd.Series(nx.pagerank(G, alpha=alpha, weight="weight"))
    pr_gravity.index = pr_gravity.index.astype(str)

    common_pr = pr_gravity.index.intersection(real_pr.index)
    log_real = np.log(real_pr.loc[common_pr]).sort_values(ascending=False)
    log_grav = np.log(pr_gravity.loc[common_pr]).loc[log_real.index]

    r2_grav = np.corrcoef(log_real, log_grav)[0, 1] ** 2
    rho_grav, _ = spearmanr(log_real, log_grav)
    r2_grav_excl = np.corrcoef(log_real.iloc[top_k_exclude:],
                               log_grav.iloc[top_k_exclude:])[0, 1] ** 2

    result = {
        "label": label, "beta_log_dist": beta,
        "n_destination_fe": len(common_nu), "r2_destination_fe": r2_nu,
        "n_gravity_pagerank": len(common_pr), "r2_gravity_pagerank": r2_grav,
        "spearman_gravity_pagerank": rho_grav,
        f"r2_gravity_pagerank_excl_top{top_k_exclude}": r2_grav_excl,
    }
    print(f"{label}: R^2(log PageRank ~ destination FE alone) = {r2_nu:.3f} (n={len(common_nu)})")
    print(f"{label}: R^2(log real PR, log gravity-implied PR) = {r2_grav:.3f}, "
          f"Spearman rho = {rho_grav:.3f} (n={len(common_pr)})")
    print(f"{label}: same R^2 excluding top {top_k_exclude} PageRank nodes = {r2_grav_excl:.3f}")
    return result


def gini(x) -> float:
    """
    Gini coefficient of a non-negative array, ignoring NaN values.

    Parameters
    ----------
    x : array-like

    Returns
    -------
    float in [0, 1], or np.nan if input is empty after NaN removal.

    Raises
    ------
    ValueError if any non-NaN value is negative.
    """
    x = np.array(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan
    if np.amin(x) < 0:
        raise ValueError("Gini requires non-negative values.")
    x_sorted = np.sort(x)
    n = len(x_sorted)
    cumulative = np.cumsum(x_sorted)
    return (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n


# ──────────────────────────────────────────────────────────
# Spatial / network dependence helpers (WP6)
# ──────────────────────────────────────────────────────────

def build_cbsa_geometry(cbsa_order, data_loaded: str = "../data_loaded",
                        crs: str = "EPSG:5070") -> gpd.GeoDataFrame:
    """
    Build CBSA polygons by dissolving counties, reindexed to `cbsa_order`.

    Reproduces the dissolve pattern used in 2_pagerank.ipynb (counties
    merged to CBSA via the OMB crosswalk, then dissolved by CBSA code),
    reprojected to a projected/metric CRS (default EPSG:5070, Albers
    Equal-Area CONUS) so that downstream centroids and distances are
    computed in meters rather than degrees.

    Parameters
    ----------
    cbsa_order : list
        CBSA ids (e.g. df['CBSA'].tolist()) defining the exact row order
        the returned GeoDataFrame must match -- this order is what any
        spatial weights matrix built on the result will be keyed by.
    data_loaded : str
    crs : str
        Target projected CRS. Must not be geographic.

    Returns
    -------
    GeoDataFrame indexed by 'CBSA', in `cbsa_order` order, with a 'name'
    column and dissolved polygon geometry.

    Raises
    ------
    KeyError if any id in `cbsa_order` has no matching CBSA polygon.
    """
    counties = load_counties(data_loaded=data_loaded)
    cbsa = load_cbsa(path=f"{data_loaded}/cbsa_2023.xlsx")

    counties = counties.merge(
        cbsa[["CBSA Code", "CBSA Title", "STCOFIPS"]], on="STCOFIPS", how="left"
    )
    cbsa_gdf = counties.dissolve(by="CBSA Code", as_index=False)
    cbsa_gdf = cbsa_gdf.rename(columns={"CBSA Title": "name", "CBSA Code": "CBSA"})
    cbsa_gdf["CBSA"] = cbsa_gdf["CBSA"].astype(str)
    cbsa_gdf = cbsa_gdf.to_crs(crs)

    missing = set(map(str, cbsa_order)) - set(cbsa_gdf["CBSA"])
    if missing:
        raise KeyError(f"No CBSA polygon found for {len(missing)} id(s): {sorted(missing)[:10]}...")

    cbsa_gdf = cbsa_gdf.set_index("CBSA").loc[[str(c) for c in cbsa_order]].reset_index()
    return cbsa_gdf


def cbsa_centroid_xy(cbsa_gdf: gpd.GeoDataFrame) -> np.ndarray:
    """
    Return an (N, 2) array of (x, y) centroids, same row order as input.

    Raises
    ------
    ValueError if `cbsa_gdf.crs` is geographic (lat/lon degrees), since
    centroids and downstream distances would then be meaningless.
    """
    if cbsa_gdf.crs is None or cbsa_gdf.crs.is_geographic:
        raise ValueError(
            "cbsa_gdf must be in a projected (metric) CRS before taking centroids; "
            f"got {cbsa_gdf.crs}."
        )
    centroids = cbsa_gdf.geometry.centroid
    return np.column_stack([centroids.x.values, centroids.y.values])


def build_cbsa_weights(cbsa_gdf: gpd.GeoDataFrame, cbsa_order, distance_band_km: float = 200) -> dict:
    """
    Build Queen-contiguity and distance-band spatial weights for CBSAs.

    Both weights objects are keyed/reindexed to exactly `cbsa_order`, so
    `w.id_order == list(cbsa_order)` for both -- this is what lets Moran's I
    be computed directly against a residual vector in the same order.

    Parameters
    ----------
    cbsa_gdf : GeoDataFrame
        Output of build_cbsa_geometry(cbsa_order), i.e. already in
        `cbsa_order` row order and a projected CRS.
    cbsa_order : list
    distance_band_km : float
        Distance-band cutoff in kilometers (converted to meters internally;
        assumes `cbsa_gdf.crs` is metric).

    Returns
    -------
    dict with keys "queen" and "distance_band", each a (W, island_ids) tuple.
    """
    cbsa_order = [str(c) for c in cbsa_order]
    gdf_indexed = cbsa_gdf.set_index("CBSA").loc[cbsa_order]

    w_queen = libpysal.weights.Queen.from_dataframe(gdf_indexed, use_index=True)
    w_db = libpysal.weights.DistanceBand.from_dataframe(
        gdf_indexed, threshold=distance_band_km * 1000, binary=True, use_index=True
    )

    assert list(w_queen.id_order) == cbsa_order, "Queen weights id_order does not match cbsa_order"
    assert list(w_db.id_order) == cbsa_order, "DistanceBand weights id_order does not match cbsa_order"

    islands_queen = list(w_queen.islands)
    islands_db = list(w_db.islands)
    if islands_queen:
        print(f"Queen contiguity: {len(islands_queen)} island CBSA(s) with zero neighbors: {islands_queen}")
    if islands_db:
        print(f"Distance band ({distance_band_km}km): {len(islands_db)} island CBSA(s) with zero neighbors: {islands_db}")

    return {"queen": (w_queen, islands_queen), "distance_band": (w_db, islands_db)}


def build_county_geometry(county_order, data_loaded: str = "../data_loaded",
                          crs: str = "EPSG:5070") -> gpd.GeoDataFrame:
    """
    Load county polygons, reprojected and reindexed to `county_order`.

    County-level analogue of build_cbsa_geometry -- no dissolve is needed
    here since counties are already the native polygon unit (unlike CBSAs,
    which are dissolved up from counties).

    Parameters
    ----------
    county_order : list
        County ids (e.g. df['STCOFIPS'].tolist()) defining the exact row
        order the returned GeoDataFrame must match.
    data_loaded : str
    crs : str
        Target projected CRS. Must not be geographic.

    Returns
    -------
    GeoDataFrame indexed by 'STCOFIPS', in `county_order` order, with
    dissolved polygon geometry.

    Raises
    ------
    KeyError if any id in `county_order` has no matching county polygon.
    """
    counties = load_counties(data_loaded=data_loaded).to_crs(crs)
    counties["STCOFIPS"] = counties["STCOFIPS"].astype(str)

    missing = set(map(str, county_order)) - set(counties["STCOFIPS"])
    if missing:
        raise KeyError(f"No county polygon found for {len(missing)} id(s): {sorted(missing)[:10]}...")

    counties = counties.set_index("STCOFIPS").loc[[str(c) for c in county_order]].reset_index()
    return counties


def build_county_weights(county_gdf: gpd.GeoDataFrame, county_order, distance_band_km: float = 200) -> dict:
    """
    Build Queen-contiguity and distance-band spatial weights for counties.

    County-level analogue of build_cbsa_weights, keyed on 'STCOFIPS'
    instead of 'CBSA'. Both weights objects are keyed/reindexed to exactly
    `county_order`, so `w.id_order == list(county_order)` for both.

    Parameters
    ----------
    county_gdf : GeoDataFrame
        Output of build_county_geometry(county_order), i.e. already in
        `county_order` row order and a projected CRS.
    county_order : list
    distance_band_km : float
        Distance-band cutoff in kilometers (converted to meters internally;
        assumes `county_gdf.crs` is metric). Counties are much smaller and
        denser than CBSAs, so a 200km band yields substantially more
        neighbors per unit than the CBSA version.

    Returns
    -------
    dict with keys "queen" and "distance_band", each a (W, island_ids) tuple.
    """
    county_order = [str(c) for c in county_order]
    gdf_indexed = county_gdf.set_index("STCOFIPS").loc[county_order]

    w_queen = libpysal.weights.Queen.from_dataframe(gdf_indexed, use_index=True)
    w_db = libpysal.weights.DistanceBand.from_dataframe(
        gdf_indexed, threshold=distance_band_km * 1000, binary=True, use_index=True
    )

    assert list(w_queen.id_order) == county_order, "Queen weights id_order does not match county_order"
    assert list(w_db.id_order) == county_order, "DistanceBand weights id_order does not match county_order"

    islands_queen = list(w_queen.islands)
    islands_db = list(w_db.islands)
    if islands_queen:
        print(f"Queen contiguity: {len(islands_queen)} island county(ies) with zero neighbors: {islands_queen}")
    if islands_db:
        print(f"Distance band ({distance_band_km}km): {len(islands_db)} island county(ies) with zero neighbors: {islands_db}")

    return {"queen": (w_queen, islands_queen), "distance_band": (w_db, islands_db)}


def morans_i_report(resid: pd.Series, w, island_ids, label: str = "",
                    permutations: int = 1000, seed: int = 42) -> dict:
    """
    Compute Moran's I on `resid` under weights `w`, dropping island ids.

    Parameters
    ----------
    resid : Series
        Residuals indexed by CBSA id, covering (at least) w.id_order.
    w : libpysal.weights.W
    island_ids : list
        Ids with zero neighbors under `w`; excluded from both `resid` and
        `w` before computing Moran's I (esda cannot handle islands).
    label : str
        Printed label for this weight scheme (e.g. "Queen contiguity").
    permutations : int
        Number of conditional permutations esda uses for `p_sim`.
    seed : int

    Returns
    -------
    dict with I, p_sim, p_norm, z_sim, n_islands_dropped, n_used.
    """
    keep_ids = [i for i in w.id_order if i not in set(island_ids)]
    w_sub = libpysal.weights.w_subset(w, ids=keep_ids)
    w_sub.transform = "r"

    resid_kept = resid.loc[keep_ids].values

    np.random.seed(seed)
    moran = esda.moran.Moran(resid_kept, w_sub, permutations=permutations)

    out = {
        "label": label,
        "I": moran.I,
        "p_sim": moran.p_sim,
        "p_norm": moran.p_norm,
        "z_sim": moran.z_sim,
        "n_islands_dropped": len(island_ids),
        "n_used": len(keep_ids),
    }
    print(f"Moran's I [{label}]: I={out['I']:.4f}  p_sim={out['p_sim']:.4f}  "
          f"p_norm={out['p_norm']:.4f}  (N={out['n_used']}, {out['n_islands_dropped']} islands dropped)")
    return out


def moran_scatterplot(resid: pd.Series, w, island_ids, label: str = "",
                      ax=None, figsize=(5, 5)) -> dict:
    """
    Moran scatterplot: standardized residual vs. its spatial lag, to see
    the clustering pattern (HH/LL/HL/LH quadrants) behind a significant
    Moran's I -- same inputs/alignment convention as `morans_i_report`.

    Parameters
    ----------
    resid : Series
        Residuals indexed by CBSA id, covering (at least) w.id_order.
    w : libpysal.weights.W
    island_ids : list
        Ids with zero neighbors under `w`; dropped before plotting.
    label : str
        Title / printed label for this weight scheme.
    ax : matplotlib Axes, optional

    Returns
    -------
    dict with the fitted Moran's I slope and the (z, lag_z) arrays plotted.
    """
    keep_ids = [i for i in w.id_order if i not in set(island_ids)]
    w_sub = libpysal.weights.w_subset(w, ids=keep_ids)
    w_sub.transform = "r"

    resid_kept = resid.loc[keep_ids].values
    z = (resid_kept - resid_kept.mean()) / resid_kept.std()
    lag_z = libpysal.weights.lag_spatial(w_sub, z)

    slope, intercept = np.polyfit(z, lag_z, 1)

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(z, lag_z, s=18, alpha=0.6, edgecolor="none")
    xs = np.array([z.min(), z.max()])
    ax.plot(xs, slope * xs + intercept, color="firebrick", linewidth=1.5,
            label=f"slope (Moran's I) = {slope:.3f}")
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_xlabel("residual (standardized)")
    ax.set_ylabel("spatial lag of residual (standardized)")
    ax.set_title(label)
    ax.legend(loc="best", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if standalone:
        plt.show()

    return {"label": label, "slope": slope, "z": z, "lag_z": lag_z}


def conley_hac_se(X: np.ndarray, resid: np.ndarray, coords_xy: np.ndarray, cutoff: float,
                  kernel: str = "uniform", param_names=None) -> dict:
    """
    Hand-coded Conley (1999) spatial-HAC sandwich covariance.

    V = (X'X)^-1 (X' Omega X) (X'X)^-1, where
    Omega = diag(resid) K diag(resid), K_ij = kernel(dist_ij, cutoff).
    No small-sample degrees-of-freedom correction is applied (matches the
    convention used by the one off-the-shelf Python implementation of this
    estimator, econtools.metrics.core.vce_shac, checked during development).

    Parameters
    ----------
    X : (N, K) ndarray
        Design matrix (e.g. ols_result.model.exog).
    resid : (N,) ndarray
    coords_xy : (N, 2) ndarray
        Projected (metric) centroid coordinates, same row order as X/resid.
    cutoff : float
        Distance cutoff in the same units as coords_xy (e.g. meters).
    kernel : str
        "uniform" (binary 0/1 within cutoff -- the default, chosen to match
        the binary distance-band spatial weights used elsewhere in this
        subsection) or "bartlett" (triangular decay to zero at cutoff).
    param_names : list, optional

    Returns
    -------
    dict with "se" (Series indexed by param_names if given, else ndarray),
    "V" (full covariance matrix).
    """
    D = np.linalg.norm(coords_xy[:, None, :] - coords_xy[None, :, :], axis=-1)

    if kernel == "uniform":
        K = (D <= cutoff).astype(float)
    elif kernel == "bartlett":
        K = np.clip(1 - D / cutoff, 0, None) * (D <= cutoff)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    XU = X * resid[:, None]
    B = XU.T @ K @ XU
    bread = np.linalg.inv(X.T @ X)
    V = bread @ B @ bread
    se = np.sqrt(np.diag(V))

    if param_names is not None:
        se = pd.Series(se, index=param_names)
    return {"se": se, "V": V}


def conley_hac_from_model(ols_result, coords_xy: np.ndarray, cutoff: float,
                          kernel: str = "uniform") -> dict:
    """
    Convenience wrapper: Conley spatial-HAC SEs for an already-fit OLS model.

    Reuses `ols_result`'s own point estimates (params) -- only the
    covariance matrix is recomputed, no refit is performed.

    Parameters
    ----------
    ols_result : statsmodels RegressionResults
    coords_xy : (N, 2) ndarray
        Same row order as the data `ols_result` was fit on.
    cutoff : float
    kernel : str

    Returns
    -------
    dict with "params", "se" (Series), "z", "p", "conf_int" (DataFrame).
    """
    X = np.asarray(ols_result.model.exog)
    resid = np.asarray(ols_result.resid)
    param_names = ols_result.model.exog_names

    conley = conley_hac_se(X, resid, coords_xy, cutoff, kernel=kernel, param_names=param_names)
    se = conley["se"]
    params = ols_result.params

    z = params / se
    p = 2 * (1 - norm.cdf(np.abs(z)))
    ci_low = params - 1.96 * se
    ci_high = params + 1.96 * se

    return {
        "params": params,
        "se": se,
        "z": z,
        "p": pd.Series(p, index=param_names),
        "conf_int": pd.DataFrame({"low": ci_low, "high": ci_high}),
    }


def cbsa_to_state_cluster(counties: pd.DataFrame, cbsa_col: str = "CBSA Code",
                          state_col: str = "STATEFP", area_col: str = "ALAND") -> pd.Series:
    """
    Assign each CBSA to its dominant (largest-land-area) member state.

    CBSAs can span multiple states, unlike counties (which nest in exactly
    one state) -- this is a documented approximation for state-clustered
    SEs at the CBSA level, not an exact nesting. The printed diagnostic
    quantifies how often the approximation actually bites (i.e. how many
    CBSAs are genuinely multi-state).

    Parameters
    ----------
    counties : DataFrame
        Must already carry the CBSA crosswalk column `cbsa_col` (e.g. via
        a merge against load_cbsa()), plus `state_col` and `area_col`
        (both present on load_counties()'s output).
    cbsa_col, state_col, area_col : str

    Returns
    -------
    Series indexed by CBSA id -> 2-digit state FIPS string.
    """
    by_state = (
        counties.dropna(subset=[cbsa_col])
        .groupby([cbsa_col, state_col])[area_col]
        .sum()
        .reset_index()
    )
    totals = by_state.groupby(cbsa_col)[area_col].sum().rename("total_aland")
    by_state = by_state.merge(totals, on=cbsa_col)
    by_state["share"] = by_state[area_col] / by_state["total_aland"]

    dominant = by_state.loc[by_state.groupby(cbsa_col)["share"].idxmax()]

    n_multistate = (dominant["share"] < 1.0).sum()
    print(f"State-cluster assignment: {n_multistate}/{len(dominant)} CBSAs are genuinely "
          f"multi-state (dominant state's land-area share < 100%).")

    return dominant.set_index(cbsa_col)[state_col]


def coef_comparison_table(coef: pd.Series, se_dict: dict, param_order=None) -> pd.DataFrame:
    """
    Side-by-side coefficient table across multiple SE methods.

    Parameters
    ----------
    coef : Series
        Point estimates, shared across all methods (e.g. model_5.params).
    se_dict : dict of {method_label: Series}
        Each Series indexed identically to `coef`.
    param_order : list, optional
        Row order; defaults to coef.index order.

    Returns
    -------
    DataFrame, one row per covariate, with (coef, se, p, stars) columns
    per method.
    """
    if param_order is None:
        param_order = list(coef.index)

    rows = []
    for param in param_order:
        row = {"param": param, "coef": coef.get(param, np.nan)}
        for method, se in se_dict.items():
            se_val = se.get(param, np.nan)
            z = row["coef"] / se_val if se_val and not np.isnan(se_val) else np.nan
            p = 2 * (1 - norm.cdf(np.abs(z))) if not np.isnan(z) else np.nan
            row[f"{method}_se"] = se_val
            row[f"{method}_p"] = p
            row[f"{method}_stars"] = make_stars(p) if not np.isnan(p) else ""
        rows.append(row)

    return pd.DataFrame(rows).set_index("param")
