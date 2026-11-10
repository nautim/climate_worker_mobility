# Climate Brain Drain

Code and aggregated data for **“Natural hazard risk is associated with lower centrality in high-skill mobility networks in the United States.”**

**Authors:** Timur Naushirvanov, Elisa Omodei, Márton Karsai, Takahiro Yabe

The analysis builds two directed, weighted mobility networks between US core-based statistical areas (CBSAs) over 2011-2022 (one of high-skill job-to-job moves derived from résumé records, one of general population moves from IRS county-to-county migration data) and compares each area's position in the two networks against local natural hazard risk.

Python handles data processing, network construction, regression and plotting; R handles the causal forest estimation via `grf`.

## Contents

- [Repository layout](#repository-layout)
- [The pipeline](#the-pipeline)
- [Robustness and supplementary analyses](#robustness-and-supplementary-analyses)
- [Installation](#installation)
- [Running the analysis](#running-the-analysis)
- [Obtaining the external data](#obtaining-the-external-data)
- [Data availability](#data-availability)
- [Code availability](#code-availability)
- [Use of generative AI](#use-of-generative-ai)
- [License](#license)

## Repository layout

```
.
├── notebooks_main/              Python pipeline, run in numbered order
│   ├── 0_data_processing.ipynb
│   ├── 1_network_creation_cbsa.ipynb
│   ├── 2_pagerank.ipynb
│   ├── 3_modelling_pr_cbsa.ipynb
│   ├── 3_modelling_pr_county.ipynb
│   ├── 4_causal_inference.ipynb
│   ├── 5_projections.ipynb
│   ├── 7_representativeness_and_validation.ipynb
│   ├── figures_final.ipynb
│   └── utils.py                 shared loaders, spatial weights, SE estimators
├── scripts_causal_forest/       grf estimation, called between notebooks 3 and 4
├── scripts_robustness/          alternative-outcome cross-checks
├── data_loaded/                 external inputs (largest are downloaded, not versioned)
├── data_generated/              outputs of the pipeline; sufficient to reproduce from step 2 on
│   ├── network_metrics/         flow tables, PageRank and HITS scores
│   └── cf_results/              causal forest estimates
├── fig/                         generated figures
├── requirements.txt
└── LICENSE
```

## The pipeline

Steps run in order, each depending on the one before. `data_generated/` is included in the repository, so **everything from step 2 onward reproduces without the licensed source data**; steps 0 and 1 need the résumé microdata, which cannot be redistributed.

`0_data_processing.ipynb` reads the raw employment histories from SQLite, maps job titles to O\*NET SOC codes, job zones and seniority levels, geocodes employers to counties, and assembles career trajectories. Writes `experience.parquet`.

`1_network_creation_cbsa.ipynb` turns job transitions and IRS migration records into directed county- and CBSA-level flow networks for 2011-2022, keeping only moves into job zone 4-5 occupations. Work counties are probabilistically reassigned to likely home counties using LODES commuting flows before aggregation. Also builds the subgroup networks (by seniority, by SOC field), the three-year windows, and the pooled 2011-2019 network. Writes flow tables and centrality scores to `data_generated/network_metrics/`.

`2_pagerank.ipynb` classifies CBSAs and counties as brain, mass, dual or peripheral hubs from the log-ratio of their high-skill and IRS PageRank, tracks how those positions shift across three-year windows, and draws the maps and correlation heatmaps.

`3_modelling_pr_cbsa.ipynb` fits the main cross-sectional model: PageRank log-differences on hazard risk and five blocks of socioeconomic controls, with a Shapley (LMG) decomposition of R², marginal-effect plots and correlation diagnostics. Also produces the spatial-dependence diagnostics (Moran's I, Conley spatial-HAC, approximate state clustering) and the hazard-measure and outcome-measure robustness tables. `3_modelling_pr_county.ipynb` repeats the model at native county resolution.

`4_causal_inference.ipynb` reads the `grf` output and plots the treatment-effect distributions, subgroup estimates, hazard-specific estimates and variable importance. Run the R scripts below first.

`5_projections.ipynb` combines the continuous-treatment estimates with FEMA National Risk Index projections to 2100 to map implied shifts in relative centrality. These are illustrative scenarios conditional on the estimated relationship holding, not forecasts.

`figures_final.ipynb` assembles the main-text figure panels, and fits the PPML gravity models used for the distance-decay results.

### The R step

Between notebooks 3 and 4, from the repository root:

```bash
Rscript scripts_causal_forest/loop_causal_forest.R        # binary treatment, per hazard
Rscript scripts_causal_forest/causal_forest_continuous.R  # continuous treatment
Rscript scripts_causal_forest/causal_forest_placebo.R     # permutation null
```

`loop_causal_forest.R` estimates heterogeneous treatment effects for each of the 18 National Risk Index hazard types plus the composite score (two hazards have too few exposed CBSAs to fit, so 16 are reported), trimming propensity scores to 0.05–0.95 and reporting ATE, ATT, ATC and ATO alongside calibration and RATE diagnostics. `causal_forest_continuous.R` estimates average marginal effects of continuous exposure under the partial-linear framework. `causal_forest_placebo.R` builds a null distribution by permuting treatment labels and refitting. Outputs land in `data_generated/cf_results/`.

## Robustness and supplementary analyses

Each of these reruns part of the pipeline under a different specification; outputs go to subdirectories of `data_generated/cf_results/`.

| Script | What changes |
| --- | --- |
| `loop_causal_forest_1119.R`, `causal_forest_continuous_1119.R`, `causal_forest_placebo_1119.R` | Pre-pandemic sample only (2011-2019) |
| `loop_causal_forest_clustered.R`, `causal_forest_continuous_clustered.R`, `causal_forest_placebo_clustered.R` | Standard errors clustered by state |
| `loop_causal_forest_county.R`, `causal_forest_continuous_county.R`, `causal_forest_placebo_county.R` | County rather than CBSA resolution |
| `hazard_measure_comparison.R`, `hazard_measure_comparison_county.R` | Composite risk score vs capacity-purged expected annual loss vs a climate-only composite |
| `perhazard_eal_comparison.R` | The 13 climate-relevant hazards' expected annual loss, one at a time |
| `scripts_robustness/outcomes_x_hazard_measures.py` | Six outcome measures crossed with three hazard measures, by OLS |
| `scripts_robustness/causal_forest_alt_outcomes.R` | The same cross, by causal forest |

`7_representativeness_and_validation.ipynb` benchmarks the résumé sample against BLS OES, ACS and County Business Patterns, and validates the resulting flows against Census LEHD Job-to-Job data.

## Installation

```bash
git clone https://github.com/nautim/climate_worker_mobility.git
cd climate_worker_mobility

conda create -n climate_worker_mobility -c conda-forge python=3.12 r-base=4.4 jupyterlab
conda activate climate_worker_mobility

pip install -r requirements.txt
Rscript -e "install.packages(c('grf', 'dplyr'), repos='https://cloud.r-project.org')"
```

Installation took about 10 minutes in testing, most of it compiling `grf` on osx-arm64.

Tested on Linux (x86_64, SUSE Linux Enterprise Server 15 SP6) and macOS 26.4.1 (Apple Silicon, MacBook Air M2), with Python 3.12.0 and R 4.4-4.5. No special hardware is needed.

## Running the analysis

```bash
jupyter-lab notebooks_main/2_pagerank.ipynb
```

Run the notebooks in numbered order in one Jupyter Lab session, pausing after `3_modelling_pr_cbsa.ipynb` for the three R scripts above. Notebook paths are relative to `notebooks_main/`, so run them from there. Expect roughly 15 minutes end to end on a personal computer, excluding installation and the causal forest scripts.

## Obtaining the external data

Small reference inputs are versioned in `data_loaded/`. The bulk files below are public but too large to keep in git, so download them into the paths shown before running the steps that need them. Every other step runs without them.

| Path | Source | Needed by |
| --- | --- | --- |
| `data_loaded/LODES/` | [LEHD LODES](https://lehd.ces.census.gov/data/) origin–destination files, aggregated by county | notebook 1 |
| `data_loaded/county/` | [Census cartographic county boundaries](https://www.census.gov/programs-surveys/geography/guidance/tiger-data-products-guide.html), 2018 and 2024 vintages | `utils.load_counties`, notebooks 2–5 |
| `data_loaded/jobpostings_aggregated/` | Job postings aggregated by county, month and NAICS | notebook 3 |
| `data_loaded/NRI_Shapefile_Counties/` | [FEMA National Risk Index](https://hazards.fema.gov/nri/) county shapefile | per-hazard EAL scores in notebook 3 |
| `data_loaded/external_validation/lehd_j2j/`, `j2j_*.csv` | [Census LEHD Job-to-Job flows](https://lehd.ces.census.gov/data/j2j.html) | notebook 7 |
| `data_loaded/data_final/` archives | ACS, SAIPE, BEA and County Health Rankings source archives | only to rebuild the `*_combined.parquet` files, which are versioned |

## Data availability

The résumé microdata come from People Data Labs via Dewey and were used under licence. They cannot be redistributed; access can be requested at [deweydata.io](https://www.deweydata.io/data-partners/people-data-labs). IRS migration data are public at [irs.gov](https://www.irs.gov/statistics/soi-tax-stats-migration-data), American Community Survey tables at [data.census.gov](https://data.census.gov/), and the remaining public sources are listed above. Variable definitions and sources are given in the paper's Supplementary Information.

The aggregated tables in `data_generated/` are sufficient to reproduce every reported result from the PageRank stage onward.

## Code availability

All analysis code is in this repository. A step-by-step workflow, dependency list and data requirements are given above.

## Use of generative AI

Generative AI tools were used for code review, debugging, and documentation. The analytical design, methodological choices and interpretation of results are the authors' own, and all outputs were checked against the underlying data.

## License

MIT. See [LICENSE](LICENSE).
