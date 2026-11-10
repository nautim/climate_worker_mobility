# ==============================================================================
# CAUSAL FOREST: Continuous Treatment (Hazard Scores → PageRank Log-Difference)
# Sample-period robustness variant: pooled 2011-2019 sample (vs. main 2011-2022 script)
# ==============================================================================
# Treats each hazard score as a continuous treatment variable.
# The estimand is the average marginal effect of one-unit change.
#
# With continuous W, grf uses the Robinson (1988) partial linear decomposition:
#   Y - E[Y|X] = tau(X) * (W - E[W|X]) + epsilon
# where tau(X) is the heterogeneous marginal effect and ATE = E[tau(X)].
#
# Note: ATT/ATC are not defined for continuous treatments.
#
# Outputs:
#   - AME SEs and p-values
#   - Calibration test (heterogeneity validity)
#   - RATE test (targeting value of CATEs)
#   - cf_continuous_summary.csv
# ==============================================================================

library(grf)
library(dplyr)

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------

DATA_PATH    <- "data_generated/df_merged_cbsa2_1119.csv"
OUTPUT_DIR   <- "data_generated/cf_results/1119"
NUM_TREES_CF <- 2000
NUM_TREES_NF <- 1000
MIN_OBS      <- 50
SEED         <- 42

set.seed(SEED)

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
# ------------------------------------------------------------------------------
# DATA
# ------------------------------------------------------------------------------

df <- read.csv(DATA_PATH)
df[df == -9999] <- NA

# Outcome
Y <- df$pagerank_logdiff

# Covariates (controls / effect modifiers)
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

# Hazards to analyse
hazards <- c(
  "AVLN_RISKS", "CFLD_RISKS", "CWAV_RISKS", "DRGT_RISKS", "ERQK_RISKS",
  "HAIL_RISKS", "HWAV_RISKS", "HRCN_RISKS", "ISTM_RISKS", "LNDS_RISKS",
  "LTNG_RISKS", "RFLD_RISKS", "SWND_RISKS", "TRND_RISKS", "TSUN_RISKS",
  "VLCN_RISKS", "WFIR_RISKS", "WNTW_RISKS", "RISK_SCORE"
)

# ------------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------------

p_to_stars <- function(p) {
  if (is.na(p))  return("")
  if (p < 0.001) return("***")
  if (p < 0.01)  return("**")
  if (p < 0.05)  return("*")
  if (p < 0.1)   return(".")
  ""
}

fmt_effect <- function(est, se, digits = 4) {
  if (is.na(est) || is.na(se) || se == 0) return(NA_character_)
  p <- 2 * (1 - pnorm(abs(est / se)))
  sprintf("%.*f%s (SE=%.4f, p=%.3f)", digits, est, p_to_stars(p), se, p)
}

# Extract differential.forest.prediction row from test_calibration output
extract_cal <- function(cal_res) {
  txt  <- utils::capture.output(print(cal_res))
  line <- txt[grepl("^\\s*differential\\.forest\\.prediction\\b", txt)]
  if (length(line) == 0) return(list(est = NA_real_, se = NA_real_, p = NA_real_))
  nums <- as.numeric(regmatches(
    line[1], gregexpr("[-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?", line[1])
  )[[1]])
  if (length(nums) < 4) return(list(est = NA_real_, se = NA_real_, p = NA_real_))
  list(est = nums[1], se = nums[2], p = nums[4])
}


# Extract RATE estimate and p-value from rank_average_treatment_effect output
extract_rate <- function(rate_res) {
  tryCatch({
    est <- rate_res$estimate
    se  <- rate_res$std.err
    list(est = est, se = se, p = 2 * (1 - pnorm(abs(est / se))))
  }, error = function(e) list(est = NA_real_, se = NA_real_, p = NA_real_))
}

make_p <- function(est, se) {
  if (is.na(est) || is.na(se) || se == 0) return(NA_real_)
  2 * (1 - pnorm(abs(est / se)))
}

# ------------------------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------------------------

summary_all <- data.frame()
cates_all   <- data.frame()

for (hz in hazards) {
  
  sep <- paste(rep("=", 60), collapse = "")
  cat(sprintf("\n%s\nHAZARD: %s (continuous)\n%s\n", sep, hz, sep))
  
  # ── 1. Continuous treatment (raw, unscaled) ─────────────────────────────────
  # W is kept on its original 0–100 scale so the ATE is interpretable as
  # the average marginal effect of a 1-unit increase in the hazard score.
  W_raw <- df[[hz]]
  if (all(is.na(W_raw))) { cat("  SKIP: all NA.\n"); next }
  
  W_range <- range(W_raw, na.rm = TRUE)
  if (diff(W_range) == 0) { cat("  SKIP: no variation in treatment.\n"); next }
  
  valid <- !is.na(W_raw) & !is.na(Y) & complete.cases(X.cf)
  X_hz  <- X.cf[valid, , drop = FALSE]
  Y_hz  <- Y[valid]
  W_hz  <- W_raw[valid]
  idk <- if ("CBSA" %in% colnames(df)) df$CBSA[valid] else seq_len(sum(valid))
  
  if (sum(valid) < MIN_OBS) {
    cat("  SKIP: insufficient observations.\n"); next
  }
  
  cat(sprintf("  n = %d  |  W range [%.3f, %.3f]  (raw scale, AME per 1-unit increase)\n",
              sum(valid), W_range[1], W_range[2]))
  
  # ── 2. Nuisance models ──────────────────────────────────────────────────────
  # With continuous W, E[W|X] is the conditional mean (not a propensity score).
  # No binary overlap trimming — grf residualises W - E[W|X] internally.
  cat("  Fitting nuisance models...\n")
  Y_hat <- predict(regression_forest(X_hz, Y_hz, num.trees = NUM_TREES_NF, seed = SEED))$predictions
  W_hat <- predict(regression_forest(X_hz, W_hz, num.trees = NUM_TREES_NF, seed = SEED))$predictions
  
  # ── 3. Causal forest ────────────────────────────────────────────────────────
  cat("  Fitting causal forest...\n")
  cf <- causal_forest(
    X_hz, Y_hz, W_hz,
    Y.hat     = Y_hat,
    W.hat     = W_hat,
    num.trees = NUM_TREES_CF,
    seed      = SEED
  )
  
  # ── 4. Average marginal effects ─────────────────────────────────────────────
  # ATE = average marginal effect across all units
  # ATO = overlap-weighted; downweights units where W ≈ E[W|X] (little
  #       identifying variation), so most credible estimate for continuous W
  ate_r <- tryCatch(
    as.list(average_treatment_effect(cf, target.sample = "all")),
    error = function(e) list(estimate = NA_real_, std.err = NA_real_)
  )
  ato_r <- tryCatch(
    as.list(average_treatment_effect(cf, target.sample = "overlap")),
    error = function(e) list(estimate = NA_real_, std.err = NA_real_)
  )
  
  cat("\n  ── Average Marginal Effect (per 1-unit increase in hazard score) ──\n")
  cat(sprintf("  ATE : %s\n", fmt_effect(ate_r$estimate, ate_r$std.err)))
  cat(sprintf("  ATO : %s\n", fmt_effect(ato_r$estimate, ato_r$std.err)))
  
  # ── 5. Calibration test ──────────────────────────────────────────────────────
  cat("\n  ── Calibration Test (CATE validity) ──\n")
  cal_res <- tryCatch(
    test_calibration(cf),
    error = function(e) { cat("  Calibration failed.\n"); NULL }
  )
  if (!is.null(cal_res)) {
    print(cal_res)
    cal <- extract_cal(cal_res)
  } else {
    cal <- list(est = NA_real_, se = NA_real_, p = NA_real_)
  }
  
  # ── 6. RATE test ─────────────────────────────────────────────────────────────
  # only supports binary treatment
  
  # ── 7. CATEs per observation ────────────────────────────────────────────────
  pred_cf   <- predict(cf, estimate.variance = TRUE)
  cate_preds <- as.numeric(pred_cf$predictions)
  cate_vars  <- pred_cf$variance.estimates
  cate_se    <- if (!is.null(cate_vars) && length(cate_vars) == length(cate_preds)) {
    sqrt(as.numeric(cate_vars))
  } else {
    rep(NA_real_, length(cate_preds))
  }
  cate_df <- data.frame(
    hazard   = hz,
    CBSA = idk,
    cate     = cate_preds,
    cate_se  = cate_se,
    treated  = W_hz,
    ps       = W_hat
  )
  cates_all <- bind_rows(cates_all, cate_df)
  
  # ── 8. Collate summary row ───────────────────────────────────────────────────
  summary_all <- bind_rows(summary_all, data.frame(
    hazard    = hz,
    n         = sum(valid),
    W_min     = W_range[1],
    W_max     = W_range[2],
    ATE_est   = ate_r$estimate,
    ATE_se    = ate_r$std.err,
    ATE_p     = make_p(ate_r$estimate, ate_r$std.err),
    ATE_fmt   = fmt_effect(ate_r$estimate, ate_r$std.err),
    ATO_est   = ato_r$estimate,
    ATO_se    = ato_r$std.err,
    ATO_p     = make_p(ato_r$estimate, ato_r$std.err),
    ATO_fmt   = fmt_effect(ato_r$estimate, ato_r$std.err),
    cal_est   = cal$est,
    cal_se    = cal$se,
    cal_p     = cal$p,
    cal_fmt   = fmt_effect(cal$est, cal$se),
    stringsAsFactors = FALSE
  ))
  
}  # end hazard loop

# ------------------------------------------------------------------------------
# SAVE & PRINT
# ------------------------------------------------------------------------------

write.csv(summary_all, file.path(OUTPUT_DIR, "cf_continuous_summary.csv"), row.names = FALSE)
write.csv(cates_all,   file.path(OUTPUT_DIR, "cf_continuous_cates.csv"),    row.names = FALSE)

cat("\n\n", paste(rep("=", 80), collapse = ""), "\n")
cat("SUMMARY TABLE — Continuous Treatment Causal Forest\n")
cat(paste(rep("=", 80), collapse = ""), "\n\n")

if (nrow(summary_all) > 0) {
  out <- summary_all[, c("hazard", "n", "ATE_fmt", "ATO_fmt", "cal_fmt")]
  colnames(out) <- c("Hazard", "N", "AME (per 1-unit)", "ATO", "Calibration")
  print(out, row.names = FALSE)
}

cat("\n\nInterpretation notes:\n")
cat("  W is on its original scale (approx. 0–100 for most hazards).\n")
cat("  ATE = average marginal effect of a 1-unit increase in hazard score on pagerank_logdiff.\n")
cat("  ATO = overlap-weighted ATE; most credible where W has most identifying variation.\n")
cat("  ATT/ATC are not defined for continuous treatments.\n")
cat("  Calibration: significant differential.forest.prediction = non-trivial heterogeneity.\n")
cat("  RATE: significant positive result = CATEs useful for targeting/prioritisation.\n")
cat("\nSignificance: *** p<0.001  ** p<0.01  * p<0.05  . p<0.1\n")
