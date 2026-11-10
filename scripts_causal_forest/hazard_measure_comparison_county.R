# ==============================================================================
# Causal forest ATE across hazard measures -- COUNTY-LEVEL VARIANT
# ==============================================================================
# County-level analogue of hazard_measure_comparison.R (see there for the full
# CBSA-level docs). Differences:
#   - DATA_PATH / OUTPUT_PATH point at the county-level table/output folder
#   - clusters = state FIPS, passed to every forest: counties nest cleanly
#     within exactly one state, so this is an EXACT clustering variable
#     (mirrors loop_causal_forest_county.R and the state-clustered SEs used
#     for the county OLS Model 5 in 3_modelling_pr_county.ipynb)
#   - single pooled 2011-2022 period only -- no 2011-2019 companion sample
#     exists at the county level (out of scope so far, see that notebook's
#     intro note)
# ==============================================================================

library(grf)
library(dplyr)

N_THREADS <- 12

NUM_TREES_CF <- 2000
NUM_TREES_NF <- 1000
PS_TRIM_LOW  <- 0.05
PS_TRIM_HIGH <- 0.95
MIN_OBS      <- 50
SEED         <- 42

DATA_PATH  <- "data_generated/df_merged_county.csv"
OUTPUT_DIR <- "data_generated/cf_results/county"
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)

feature_cols <- c(
  "gdp_pc", "n_postings_100k", "education_index",
  "income_index_adj", "health_index_adj", "personal_income_pc",
  "POPULATION", "poverty_rate_all",
  "nat_amenity_score", "real_estate_taxes_med",
  "income_price_ratio2023",
  "emp_100k", "est_100k", "inequality_ratio",
  "pop_density", "dividends_pc"
)

measures <- c("RISK_SCORE", "EAL_SCORE", "climate_eal_meanscore")

fmt_effect <- function(est, se, digits = 5) {
  if (is.na(est) || is.na(se) || se == 0) return(NA_character_)
  p <- 2 * (1 - pnorm(abs(est / se)))
  sprintf("%.*f (SE=%.5f, p=%.4f)", digits, est, se, p)
}

set.seed(SEED)
df <- read.csv(DATA_PATH)
df[df == -9999] <- NA

Y <- df$pagerank_logdiff
X <- df[, feature_cols, drop = FALSE]
X.cf <- model.matrix(~ . - 1, data = X)

# Exact clustering variable: every county nests in exactly one state.
clusters_all <- as.integer(as.factor(substr(df$STCOFIPS, 1, 2)))

results <- data.frame()
period_label <- "2011-2022"

for (hz in measures) {

  cat(sprintf("\n[%s] %s\n", period_label, hz))

  W <- tryCatch({
    med <- median(df[[hz]], na.rm = TRUE)
    ifelse(df[[hz]] >= med, 1L, 0L)
  }, error = function(e) NULL)
  if (is.null(W)) { cat("  SKIP: could not binarize.\n"); next }

  valid       <- !is.na(W) & !is.na(Y)
  X_hz        <- X.cf[valid, , drop = FALSE]
  Y_hz        <- Y[valid]
  W_hz        <- W[valid]
  clusters_hz <- clusters_all[valid]

  if (sum(valid) < MIN_OBS || length(unique(W_hz)) < 2) {
    cat("  SKIP: insufficient data or no treatment variation.\n"); next
  }

  W_hat_mod <- tryCatch(
    regression_forest(X_hz, W_hz, num.trees = NUM_TREES_NF, seed = SEED,
                       clusters = clusters_hz, num.threads = N_THREADS),
    error = function(e) NULL
  )
  if (is.null(W_hat_mod)) { cat("  SKIP: propensity model failed.\n"); next }

  ps   <- predict(W_hat_mod)$predictions
  keep <- ps > PS_TRIM_LOW & ps < PS_TRIM_HIGH
  if (sum(keep) < MIN_OBS) { cat("  SKIP: too few after overlap trim.\n"); next }

  Xk         <- X_hz[keep, , drop = FALSE]
  Yk         <- Y_hz[keep]
  Wk         <- W_hz[keep]
  clusters_k <- clusters_hz[keep]

  Y_hat <- predict(regression_forest(Xk, Yk, num.trees = NUM_TREES_NF, seed = SEED,
                                      clusters = clusters_k, num.threads = N_THREADS))$predictions
  W_hat <- predict(regression_forest(Xk, Wk, num.trees = NUM_TREES_NF, seed = SEED,
                                      clusters = clusters_k, num.threads = N_THREADS))$predictions

  cf <- causal_forest(
    Xk, Yk, Wk,
    Y.hat = Y_hat, W.hat = W_hat,
    num.trees = NUM_TREES_CF, seed = SEED,
    clusters = clusters_k, num.threads = N_THREADS
  )

  ate_r <- as.list(average_treatment_effect(cf, target.sample = "all"))
  cat(sprintf("  n_keep=%d  ATE=%s\n", sum(keep), fmt_effect(ate_r$estimate, ate_r$std.err)))

  p_val <- 2 * (1 - pnorm(abs(ate_r$estimate / ate_r$std.err)))
  results <- bind_rows(results, data.frame(
    period    = period_label,
    hazard    = hz,
    n_total   = sum(valid),
    n_keep    = sum(keep),
    ATE_est   = ate_r$estimate,
    ATE_se    = ate_r$std.err,
    ATE_p     = p_val,
    stringsAsFactors = FALSE
  ))
}

write.csv(results, file.path(OUTPUT_DIR, "hazard_measure_comparison.csv"), row.names = FALSE)

cat("\n\n", paste(rep("=", 90), collapse = ""), "\n")
cat("SUMMARY: causal forest ATE across hazard measures (county level, 2011-2022)\n")
cat(paste(rep("=", 90), collapse = ""), "\n\n")

p_to_stars <- function(p) {
  if (is.na(p))  return("")
  if (p < 0.001) return("***")
  if (p < 0.01)  return("**")
  if (p < 0.05)  return("*")
  if (p < 0.1)   return(".")
  ""
}
results$stars <- sapply(results$ATE_p, p_to_stars)
results$ATE_fmt <- sprintf("%.5f%s (SE=%.5f, p=%.4f)", results$ATE_est, results$stars, results$ATE_se, results$ATE_p)

for (hz in measures) {
  cat(sprintf("%-28s\n", hz))
  sub <- results[results$hazard == hz, ]
  for (i in seq_len(nrow(sub))) {
    r <- sub[i, ]
    cat(sprintf("  %-10s N=%3d  ATE=%s\n", r$period, r$n_keep, r$ATE_fmt))
  }
  cat("\n")
}
