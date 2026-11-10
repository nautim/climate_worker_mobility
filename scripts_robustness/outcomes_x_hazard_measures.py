"""
Outcome measures x hazard measures cross-check: alternative outcome measures
x alternative hazard measures.

Standalone script -- run directly, no notebook kernel state required:

    conda activate phd_work   # or any env with pandas/numpy/statsmodels/scipy
    python scripts_robustness/outcomes_x_hazard_measures.py

Run from the repo root (paths below are relative to it, matching the rest of
the pipeline's convention).

What this does
---------------
Re-fits the paper's headline Model 5 covariate specification (unchanged) for
every combination of:

  - 6 dependent variables (outcome measures):
      * pagerank_logdiff        -- existing headline DV
      * instrength_logratio     -- weighted in-strength log-ratio (new)
      * outstrength_logratio    -- weighted out-strength log-ratio (new)
      * indegree_logratio       -- unweighted in-degree log-ratio (new)
      * outdegree_logratio      -- unweighted out-degree log-ratio (new)
      * net_migration_diff      -- net migration rate difference (new)

  - 3 hazard measures, all already merged into
    data_generated/df_merged_cbsa2.csv by the hazard-measure robustness check:
      * RISK_SCORE               -- existing composite (capacity-inclusive)
      * EAL_SCORE                -- full-hazard EAL (capacity-purged)
      * climate_eal_meanscore    -- climate-only, mean-of-scores composite

i.e. 6 x 3 = 18 OLS fits, all on the same 698-CBSA pooled 2011-2022 sample,
covariates held fixed. See notebooks_main/3_modelling_pr_cbsa.ipynb (the
"For SI: Alternative outcome measures" and "For SI: Capacity-purged and
climate-only hazard measures" sections) for the narrative writeup and the
original derivations these measures came from.

Prerequisites
-------------
- data_generated/df_merged_cbsa2.csv must already contain the hazard-measure
  columns (EAL_SCORE, climate_eal_meanscore). These were added in place by
  the hazard-measure robustness check; if starting from a fresh clone
  without them, rerun the "Capacity-purged and climate-only hazard measures"
  cells in notebooks_main/3_modelling_pr_cbsa.ipynb first (they read
  data_loaded/NRI_Shapefile_Counties/NRI_Shapefile_Counties.shp and write
  those columns back into the CSV).
- data_generated/network_metrics/cbsa2_flows_{exp,irs}_1122.parquet must exist
  (built by notebooks_main/1_network_creation_cbsa.ipynb).

Output
------
Prints an 18-row comparison table to stdout and saves it to
data_generated/cf_results/outcomes_x_hazard_measures.csv

Also saves data_generated/df_merged_cbsa2_alt_outcomes.csv -- the same 698-CBSA table
with the 5 new outcome columns added, consumed by
scripts_robustness/causal_forest_alt_outcomes.R for the causal-forest version
of the same 6x3 grid.
"""

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

GEN = "data_generated"
EPS = 1e-9

OUTCOME_MEASURES = {
    "PageRank log-diff (existing)": "pagerank_logdiff",
    "In-strength log-ratio": "instrength_logratio",
    "Out-strength log-ratio": "outstrength_logratio",
    "In-degree log-ratio": "indegree_logratio",
    "Out-degree log-ratio": "outdegree_logratio",
    "Net migration rate diff": "net_migration_diff",
}

HAZARD_MEASURES = {
    "RISK_SCORE (existing)": "RISK_SCORE",
    "EAL_SCORE (full-hazard EAL)": "EAL_SCORE",
    "Climate-only (mean-score)": "climate_eal_meanscore",
}

MODEL_5_COVARIATES = (
    "log_pop_centered + log_gdp_centered + log_pop_centered*log_gdp_centered + I(log_pop_centered**2)"
    "+ log_taxes_med_centered + log_dividends_centered + income_price_ratio2023_centered"
    "+ log_est_100k_centered + log_emp_100k_centered + log_n_postings_centered"
    "+ education_centered + health_centered + inequality_centered"
    "+ nat_amenity_centered + water_perc_centered"
)


def build_node_stats(flows_path: str) -> pd.DataFrame:
    """Weighted in-/out-strength, unweighted in-/out-degree per CBSA."""
    flows = pd.read_parquet(flows_path)
    in_strength = flows.groupby("CBSA Code_next")["n"].sum().rename("in_strength")
    out_strength = flows.groupby("CBSA Code")["n"].sum().rename("out_strength")
    in_degree = flows.groupby("CBSA Code_next")["CBSA Code"].nunique().rename("in_degree")
    out_degree = flows.groupby("CBSA Code")["CBSA Code_next"].nunique().rename("out_degree")
    stats = pd.concat([in_strength, out_strength, in_degree, out_degree], axis=1).fillna(0)
    stats.index.name = "CBSA"
    return stats.reset_index()


def build_outcome_measures(df: pd.DataFrame) -> pd.DataFrame:
    """Add instrength_logratio, indegree_logratio, net_migration_diff to df in place."""
    exp_stats = build_node_stats(f"{GEN}/network_metrics/cbsa2_flows_exp_1122.parquet")
    irs_stats = build_node_stats(f"{GEN}/network_metrics/cbsa2_flows_irs_1122.parquet")
    exp_stats["CBSA"] = exp_stats["CBSA"].astype(str)
    irs_stats["CBSA"] = irs_stats["CBSA"].astype(str)

    df = df.merge(exp_stats, on="CBSA", how="left", suffixes=("", "_exp"))
    df = df.rename(columns={"in_strength": "in_strength_exp", "out_strength": "out_strength_exp",
                             "in_degree": "in_degree_exp", "out_degree": "out_degree_exp"})
    df = df.merge(irs_stats, on="CBSA", how="left", suffixes=("", "_irs"))
    df = df.rename(columns={"in_strength": "in_strength_irs", "out_strength": "out_strength_irs",
                             "in_degree": "in_degree_irs", "out_degree": "out_degree_irs"})

    n_missing = df[["in_strength_exp", "in_strength_irs", "in_degree_exp", "in_degree_irs",
                    "out_strength_exp", "out_strength_irs", "out_degree_exp", "out_degree_irs"]].isna().sum().sum()
    if n_missing:
        raise ValueError(f"{n_missing} missing values introduced when merging flow stats -- check CBSA dtype alignment")

    df["instrength_logratio"] = np.log(df["in_strength_exp"] + EPS) - np.log(df["in_strength_irs"] + EPS)
    df["outstrength_logratio"] = np.log(df["out_strength_exp"] + EPS) - np.log(df["out_strength_irs"] + EPS)
    df["indegree_logratio"] = np.log(df["in_degree_exp"] + EPS) - np.log(df["in_degree_irs"] + EPS)
    df["outdegree_logratio"] = np.log(df["out_degree_exp"] + EPS) - np.log(df["out_degree_irs"] + EPS)

    net_exp = (df["in_strength_exp"] - df["out_strength_exp"]) / (df["in_strength_exp"] + df["out_strength_exp"] + EPS)
    net_irs = (df["in_strength_irs"] - df["out_strength_irs"]) / (df["in_strength_irs"] + df["out_strength_irs"] + EPS)
    df["net_migration_diff"] = net_exp - net_irs
    return df


def stars(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "."
    return ""


def main():
    df = pd.read_csv(f"{GEN}/df_merged_cbsa2.csv", index_col=0)
    df["CBSA"] = df["CBSA"].astype(str)

    missing_hazards = [c for c in HAZARD_MEASURES.values() if c not in df.columns]
    if missing_hazards:
        raise ValueError(
            f"Missing hazard columns {missing_hazards} in {GEN}/df_merged_cbsa2.csv -- "
            "rerun the 'Capacity-purged and climate-only hazard measures' cells first."
        )

    df = build_outcome_measures(df)

    for h_col in HAZARD_MEASURES.values():
        df[f"{h_col}_centered"] = df[h_col] - df[h_col].mean()

    # Persist the augmented table (original columns + the 5 new outcome measures)
    # so scripts_robustness/causal_forest_alt_outcomes.R can fit the same 6x3 grid
    # via causal forest without duplicating the flow-parquet outcome construction in R.
    df.to_csv(f"{GEN}/df_merged_cbsa2_alt_outcomes.csv", index=False)

    rows = []
    for outcome_label, outcome_col in OUTCOME_MEASURES.items():
        for hazard_label, hazard_col in HAZARD_MEASURES.items():
            term = f"{hazard_col}_centered"
            formula = f"{outcome_col} ~ {MODEL_5_COVARIATES} + {term}"
            m = smf.ols(formula=formula, data=df).fit(cov_type="HC3")
            rows.append({
                "outcome": outcome_label, "hazard": hazard_label,
                "coef": m.params[term], "se": m.bse[term], "p": m.pvalues[term],
                "N": int(m.nobs), "R2": m.rsquared,
            })

    results = pd.DataFrame(rows)
    results.to_csv(f"{GEN}/cf_results/outcomes_x_hazard_measures.csv", index=False)

    print(f"{'Outcome':32s} {'Hazard':30s} {'coef':>12s} {'p':>8s} {'N':>5s} {'R2':>6s}")
    for outcome_label in OUTCOME_MEASURES:
        sub = results[results["outcome"] == outcome_label]
        for _, r in sub.iterrows():
            print(f"{outcome_label:32s} {r['hazard']:30s} {r['coef']: 12.6f}{stars(r['p']):3s} "
                  f"{r['p']:8.4f} {int(r['N']):5d} {r['R2']:6.3f}")
        print()

    print(f"Saved to {GEN}/cf_results/outcomes_x_hazard_measures.csv")


if __name__ == "__main__":
    main()
