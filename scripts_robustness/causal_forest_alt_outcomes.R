# ==============================================================================
# Outcome measures x hazard measures cross-check, causal forest version:
# alternative outcome measures x alternative hazard measures, causal-forest ATE.
#
# Python/OLS counterpart: scripts_robustness/outcomes_x_hazard_measures.py
# (same 6 outcomes x 3 hazards = 18 fits, same covariate set, same 698-CBSA
# pooled 2011-2022 sample). This answers the same question -- does the
# outcome-measure / hazard-measure finding hold under a flexible functional
# form? -- flagged as "not yet done" in the alternative-outcome-measures
# writeup in notebooks_main/3_modelling_pr_cbsa.ipynb ("For SI: Alternative
# outcome measures").
#
# Treatment is binarised at the median for each hazard measure (matching
# loop_causal_forest.R's main-loop convention); no clustering, since CBSAs
# can span multiple states (same choice as loop_causal_forest.R).
#
# Reads data_generated/df_merged_cbsa2_alt_outcomes.csv (the augmented table the
# Python script above saves -- original df_merged_cbsa2.csv columns plus the
# 5 new outcome columns), since the R env here has no parquet/arrow
# support to rebuild them from the flow tables directly.
#
# Output: data_generated/cf_results/outcomes_x_hazard_measures_cf.csv
# ==============================================================================

library(grf)
library(dplyr)

DATA_PATH    <- "data_generated/df_merged_cbsa2_alt_outcomes.csv"
OUTPUT_PATH  <- "data_generated/cf_results/outcomes_x_hazard_measures_cf.csv"
NUM_TREES_CF <- 2000
NUM_TREES_NF <- 1000
PS_TRIM_LOW  <- 0.05
PS_TRIM_HIGH <- 0.95
MIN_OBS      <- 50
SEED         <- 42
N_THREADS    <- 12   # grf defaults to all detected cores -- capped, shared machine

set.seed(SEED)

df <- read.csv(DATA_PATH)
df[df == -9999] <- NA

feature_cols <- c(
  "gdp_pc", "n_postings_100k", "education_index",
  "income_index_adj", "health_index_adj", "personal_income_pc",
  "POPULATION", "poverty_rate_all",
  "nat_amenity_score", "real_estate_taxes_med",
  "income_price_ratio2023",
  "emp_100k", "est_100k", "inequality_ratio",
  "pop_density", "dividends_pc"
)

X    <- df[, feature_cols, drop = FALSE]
X.cf <- model.matrix(~ . - 1, data = X)

OUTCOME_MEASURES <- c(
  "PageRank log-diff (existing)" = "pagerank_logdiff",
  "In-strength log-ratio"        = "instrength_logratio",
  "Out-strength log-ratio"       = "outstrength_logratio",
  "In-degree log-ratio"          = "indegree_logratio",
  "Out-degree log-ratio"         = "outdegree_logratio",
  "Net migration rate diff"      = "net_migration_diff"
)

HAZARD_MEASURES <- c(
  "RISK_SCORE (existing)"        = "RISK_SCORE",
  "EAL_SCORE (full-hazard EAL)"  = "EAL_SCORE",
  "Climate-only (mean-score)"    = "climate_eal_meanscore"
)

p_to_stars <- function(p) {
  if (is.na(p))  return("")
  if (p < 0.001) return("***")
  if (p < 0.01)  return("**")
  if (p < 0.05)  return("*")
  if (p < 0.1)   return(".")
  ""
}

fmt_effect <- function(est, se, digits = 6) {
  if (is.na(est) || is.na(se) || se == 0) return(NA_character_)
  p <- 2 * (1 - pnorm(abs(est / se)))
  sprintf("%.*f%s (SE=%.6f, p=%.4f)", digits, est, p_to_stars(p), se, p)
}

results <- data.frame()

for (hazard_label in names(HAZARD_MEASURES)) {
  hazard_col <- HAZARD_MEASURES[[hazard_label]]

  W <- ifelse(df[[hazard_col]] >= median(df[[hazard_col]], na.rm = TRUE), 1L, 0L)

  for (outcome_label in names(OUTCOME_MEASURES)) {
    outcome_col <- OUTCOME_MEASURES[[outcome_label]]
    Y <- df[[outcome_col]]

    valid <- !is.na(W) & !is.na(Y) & complete.cases(X.cf)
    X_hz  <- X.cf[valid, , drop = FALSE]
    Y_hz  <- Y[valid]
    W_hz  <- W[valid]

    if (sum(valid) < MIN_OBS || length(unique(W_hz)) < 2) {
      cat(sprintf("SKIP: %-28s x %-28s (insufficient data)\n", outcome_label, hazard_label))
      next
    }

    W_hat_mod <- regression_forest(X_hz, W_hz, num.trees = NUM_TREES_NF, seed = SEED,
                                    num.threads = N_THREADS)
    ps   <- predict(W_hat_mod)$predictions
    keep <- ps > PS_TRIM_LOW & ps < PS_TRIM_HIGH

    if (sum(keep) < MIN_OBS) {
      cat(sprintf("SKIP: %-28s x %-28s (too few after overlap trim)\n", outcome_label, hazard_label))
      next
    }

    Xk <- X_hz[keep, , drop = FALSE]
    Yk <- Y_hz[keep]
    Wk <- W_hz[keep]

    Y_hat <- predict(regression_forest(Xk, Yk, num.trees = NUM_TREES_NF, seed = SEED,
                                        num.threads = N_THREADS))$predictions
    W_hat <- predict(regression_forest(Xk, Wk, num.trees = NUM_TREES_NF, seed = SEED,
                                        num.threads = N_THREADS))$predictions

    cf <- causal_forest(Xk, Yk, Wk, Y.hat = Y_hat, W.hat = W_hat,
                         num.trees = NUM_TREES_CF, seed = SEED,
                         num.threads = N_THREADS)

    ate <- as.list(average_treatment_effect(cf, target.sample = "all"))
    p   <- 2 * (1 - pnorm(abs(ate$estimate / ate$std.err)))

    cat(sprintf("%-28s x %-28s  %s\n", outcome_label, hazard_label,
                fmt_effect(ate$estimate, ate$std.err)))

    results <- bind_rows(results, data.frame(
      outcome   = outcome_label,
      hazard    = hazard_label,
      ATE_est   = ate$estimate,
      ATE_se    = ate$std.err,
      ATE_p     = p,
      ATE_fmt   = fmt_effect(ate$estimate, ate$std.err),
      n_total   = sum(valid),
      n_keep    = sum(keep),
      stringsAsFactors = FALSE
    ))
  }
}

dir.create(dirname(OUTPUT_PATH), recursive = TRUE, showWarnings = FALSE)
write.csv(results, OUTPUT_PATH, row.names = FALSE)
cat(sprintf("\nSaved %d rows to %s\n", nrow(results), OUTPUT_PATH))
