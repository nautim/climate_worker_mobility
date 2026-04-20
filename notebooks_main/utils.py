"""
Shared utilities for the climate_worker_mobility notebook pipeline.

Usage in each notebook (add to the imports cell):
    from utils import load_counties, load_cbsa, set_plot_style, ...
"""

import gc
import glob
import warnings

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns


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
