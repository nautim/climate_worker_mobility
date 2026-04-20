# Climate Brain Drain

This repository accompanies the paper **“Climate Brain Drain: How natural hazard risk reshapes the geography of high-skilled mobility in the United States.”**

**Authors:** Timur Naushirvanov, Elisa Omodei, Márton Karsai, Takahiro Yabe

## Table of Contents

- [Repository Structure](#repository-structure)
- [Pipeline Overview and Usage](#pipeline-overview-and-usage)
  - [0. Data Processing](#0-data-processing)
  - [1. Network Construction](#1-network-construction)
  - [2. Network Metrics and Classification](#2-network-metrics-and-classification)
  - [3. Regression Analysis](#3-regression-analysis)
  - [4. Causal Forest Estimation (in R)](#4-causal-forest-estimation-in-r)
    - [Binary Treatment](#binary-treatment)
    - [Continuous Treatment](#continuous-treatment)
    - [Placebo / Permutation Test](#placebo--permutation-test)
  - [5. Causal Inference Visualisation](#5-causal-inference-visualisation)
  - [6. Projections](#6-projections)
  - [Final Figures](#final-figures)
- [Reproducibility and Implementation](#reproducibility-and-implementation)
  - [System Requirements](#system-requirements)
  - [Computational Environment](#computational-environment)
  - [Installation](#installation)
  - [Usage](#usage)
- [Data Availability](#data-availability)
- [Code Availability](#code-availability)
- [Statement on the Use of Generative AI](#statement-on-the-use-of-generative-ai)

<hr style="height:2px;">


## Repository Structure

```markdown

.
├── LICENSE
├── README.md
├── requirements.txt
├── notebooks_main/
│   ├── 0_data_processing.ipynb
│   ├── 1_network_creation_cbsa.ipynb
│   ├── 2_pagerank.ipynb
│   ├── 3_modelling_pr_cbsa.ipynb
│   ├── 4_causal_inference.ipynb
│   ├── 5_projections.ipynb
│   ├── figures_final.ipynb
│   └── utils.py
├── scripts_causal_forest/
│   ├── loop_causal_forest.R
│   └── causal_forest_continuous.R
├── data_generated/
│   ├── network_metrics/
│   └── cf_results/
├── data_loaded/
├── data_raw/
└── fig/
    └── si/

```
<hr style="height:2px;">

## Pipeline Overview and Usage

The repository is organised as a sequential, modular pipeline, progressing from raw data processing to network construction, empirical analysis, causal inference, and projections. Individual notebooks can be executed independently, but full reproduction of results requires running preceding steps due to data dependencies. Processed datasets sufficient to reproduce the analysis from the PageRank computation stage onward are provided in `data_generated/`. Earlier steps, including network construction, require access to the original raw data and are not directly reproducible from the repository.

### 0. Data Processing  
**`0_data_processing.ipynb`**  
Loads raw employment data from a SQLite database, matches experiences to O*NET job codes, job zones, and seniority levels, geocodes company locations to counties, and constructs career trajectory features.  
**Output:** `experience.parquet`


### 1. Network Construction  
**`1_network_creation_cbsa.ipynb`**  
Builds directed worker-flow networks from job transitions and IRS tax migration data at the county and CBSA levels (2011–2022), restricted to high-skill occupations (job zone ≥ 4). Also constructs subgroup networks by seniority and SOC field, as well as three-year periods.  
**Output:** flow tables and HITS/PageRank metrics stored in `network_metrics/`


### 2. Network Metrics and Classification  
**`2_pagerank.ipynb`**  
Computes and visualises dual-hub metrics by classifying CBSAs and counties into Brain, Mass, Dual, or Peripheral hubs based on the log-ratio of High-Skill and IRS PageRank. Tracks temporal dynamics across three-year windows and produces geographic maps and correlation heatmaps.


### 3. Regression Analysis  
**`3_modelling_pr_cbsa.ipynb`**  
Estimates CBSA-level regressions of PageRank log-differences on climate risk indicators and socioeconomic controls. Includes subgroup analyses and produces coefficient plots, marginal effects, and correlation diagnostics.


### 4. Causal Forest Estimation (in R)

#### Binary Treatment  
**`loop_causal_forest.R`**  
Estimates heterogeneous treatment effects of 19 hazard types using `grf::causal_forest`, including overlap trimming and estimation of ATE, ATT, ATC, and ATO. Calibration and RATE tests are used to assess the validity and usefulness of heterogeneous treatment effects. Outputs: `cf_cates.csv`, `cf_summary.csv`, `cf_varimp.csv`, `cf_subgroups.csv`

#### Continuous Treatment  
**`causal_forest_continuous.R`**  
Estimates average marginal effects of continuous hazard exposure using the partial linear framework. Reports ATE and overlap-weighted ATO, alongside calibration diagnostics. Outputs: `cf_continuous_cates.csv`, `cf_contunuous_summary.csv`

#### Placebo / Permutation Test  
**`causal_forest_placebo.R`**  
Builds a null distribution for the estimated treatment effect by repeatedly permuting treatment labels and re-running the causal forest pipeline under the sharp null of no effect. Outputs: `cf_summary_random.csv`, `cf_placebo_summary.csv`

### 5. Causal Inference Visualisation  
**`4_causal_inference.ipynb`**  
Visualises causal forest outputs, including treatment effects, heterogeneity, and variable importance.


### 6. Projections  
**`5_projections.ipynb`**  
Combines estimated effects with projected hazard exposure to 2100 to infer future spatial shifts in high-skill worker attractiveness.

---

### Final Figures  
**`figures_final.ipynb`**  
Produces some of the publication-ready figures.

<hr style="height:2px;">

## Reproducibility and Implementation

The repository is organised as a sequential, modular pipeline, progressing from raw data processing to network construction, empirical analysis, causal inference, and projections. Individual notebooks can be executed independently, but full reproduction of results requires running preceding steps due to data dependencies. Processed datasets sufficient to reproduce the analysis from the PageRank computation stage onward are provided in `data_generated/`. Earlier steps, including network construction, require access to the original raw data and are not directly reproducible from the repository.

### System Requirements

The code was developed and tested in the following environments:

- Linux (x86_64, SUSE Linux Enterprise Server 15 SP6)
- macOS (Apple Silicon, macOS 26.4.1, MacBook Air M2)

Software versions:
- Python 3.12.0
- R 4.4–4.5 (tested across environments)

The code is expected to run on standard systems with similar configurations. No specialised hardware is required.

---
### Computational Environment

#### Python dependencies
See `requirements.txt`.

#### R dependencies
- grf  
- dplyr  

---
### Installation

Clone the repository:

```bash
git clone https://github.com/nautim/climate_worker_mobility.git
cd climate_worker_mobility
```

Create a conda environment with Python, R, and Jupyter Lab:

```bash
conda create -n climate_worker_mobility \
  -c conda-forge \
  python=3.12 \
  r-base=4.4 \
  jupyterlab
```

Activate the environment:

```bash
conda activate climate_worker_mobility
```

Install Python dependencies and R packages:

```bash
pip install -r requirements.txt

Rscript -e "install.packages(c('grf', 'dplyr'), repos='https://cloud.r-project.org')"
```

Observed installation time was approximately 10 minutes, including around 7 minutes for compiling the grf package (notably on osx-arm64).

---
### Usage

Start Jupyter Lab and open a notebook:

```bash
jupyter-lab notebooks_main/2_pagerank.ipynb
```

All further notebooks in notebooks_main/ can be executed within the same Jupyter Lab session.

After running notebook 3_modelling_pr_cbsa.ipynb abd before 4_causal_inference.ipynb, run the causal forest scripts from the terminal:

```bash
Rscript scripts_causal_forest/loop_causal_forest.R
Rscript scripts_causal_forest/causal_forest_continuous.R
Rscript scripts_causal_forest/causal_forest_placebo.R
```

Remaining notebooks can then be executed in Jupyter Lab.

Expected runtime is approximately 15 minutes on a standard personal computer (excluding installation time).

<hr style="height:2px;">

## Data Availability

Raw employment data cannot be shared due to licensing and privacy restrictions.

This repository provides processed and aggregated datasets sufficient to reproduce all analyses from the network construction stage onward. External datasets (IRS migration data, National Risk Index, census and socioeconomic indicators) are publicly available and can be obtained from their respective sources.

<hr style="height:2px;">

## Code Availability

All code required to reproduce the analysis, figures, and results is available in this repository. The workflow combines Python (data processing, network analysis, visualisation) and R (causal inference using `grf`).

<hr style="height:2px;">

## Statement on the Use of Generative AI

Generative AI tools were used in a supportive capacity during this project, primarily for code review, debugging, and improving code structure and documentation. All analytical decisions, methodological design, and interpretation of results were carried out by the authors.

No generative AI tools were used to produce original research findings, perform core analyses, or substitute domain-specific judgment. All outputs were critically assessed and validated to ensure accuracy and consistency with the research objectives.

