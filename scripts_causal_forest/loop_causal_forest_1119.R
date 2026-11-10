# ==============================================================================
# CAUSAL FOREST ANALYSIS: Disaster Risk Types → PageRank Log-Difference
# Sample-period robustness variant: pooled 2011-2019 sample (vs. main 2011-2022 script)
# ==============================================================================
# Estimates heterogeneous treatment effects of each disaster risk type
# (binarised at median) on regional centrality (pagerank_logdiff).
#
# Outputs per hazard:
#   - ATE, ATT, ATC, ATO with SEs and p-values
#   - CATE-level calibration test (Chernozhukov et al.)
#   - RATE test (Rank-Weighted Average Treatment Effect) for CATE validity
#   - Variable importance
#   - Per-observation CATEs saved to CSV
# ==============================================================================

library(grf)
library(dplyr)

# ------------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------------

DATA_PATH    <- "data_generated/df_merged_cbsa2_1119.csv"
OUTPUT_DIR   <- "data_generated/cf_results/1119"
NUM_TREES_CF <- 2000         # causal forest trees
NUM_TREES_NF <- 1000         # nuisance/propensity forest trees
PS_TRIM_LOW  <- 0.05         # propensity score overlap trim (lower)
PS_TRIM_HIGH <- 0.95         # propensity score overlap trim (upper)
MIN_OBS      <- 50           # minimum observations to proceed
SEED         <- 42

set.seed(SEED)

dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
# ------------------------------------------------------------------------------
# DATA
# ------------------------------------------------------------------------------

df <- read.csv(DATA_PATH)
df[df == -9999] <- NA         # replace missing values

# Outcome
Y <- df$pagerank_logdiff

# No clustering: CBSAs span multiple states, so state-level clustering
# is not appropriate here.

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
X.cf <- model.matrix(~ . - 1, data = X)   # one-hot encodes any factors
varnames <- colnames(X.cf)

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
  line <- line[1]
  nums <- as.numeric(regmatches(
    line, gregexpr("[-+]?[0-9]*\\.?[0-9]+(?:[eE][-+]?[0-9]+)?", line)
  )[[1]])
  if (length(nums) < 4) return(list(est = NA_real_, se = NA_real_, p = NA_real_))
  list(est = nums[1], se = nums[2], p = nums[4])
}

# Extract RATE estimate and p-value from rank_average_treatment_effect output
extract_rate <- function(rate_res) {
  tryCatch({
    est <- rate_res$estimate
    se  <- rate_res$std.err
    p   <- 2 * (1 - pnorm(abs(est / se)))
    list(est = est, se = se, p = p)
  }, error = function(e) list(est = NA_real_, se = NA_real_, p = NA_real_))
}

# ------------------------------------------------------------------------------
# RESULT CONTAINERS
# ------------------------------------------------------------------------------

summary_all <- data.frame()
cates_all   <- data.frame()
varimp_all  <- data.frame()

# ------------------------------------------------------------------------------
# MAIN LOOP
# ------------------------------------------------------------------------------

for (hz in hazards) {
  
  sep <- paste(rep("=", 60), collapse = "")
  cat(sprintf("\n%s\nHAZARD: %s\n%s\n", sep, hz, sep))
  
  # ── 1. Binarise treatment at median ────────────────────────────────────────
  W <- tryCatch({
    med <- median(df[[hz]], na.rm = TRUE)
    if (!is.finite(med)) stop("non-finite median")
    ifelse(df[[hz]] >= med, 1L, 0L)
  }, error = function(e) {
    cat("  SKIP: could not compute median —", conditionMessage(e), "\n"); NULL
  })
  if (is.null(W)) next
  
  
  valid       <- !is.na(W) & !is.na(Y)
  X_hz        <- X.cf[valid, , drop = FALSE]
  Y_hz        <- Y[valid]
  W_hz        <- W[valid]
  ids_hz      <- if ("CBSA" %in% colnames(df)) df$CBSA[valid] else seq_len(sum(valid))
  
  if (nrow(X_hz) < MIN_OBS || length(unique(W_hz)) < 2) {
    cat("  SKIP: insufficient data or no treatment variation.\n"); next
  }
  
  cat(sprintf("  Full sample n = %d  (treated=%d, control=%d)\n",
              length(W_hz), sum(W_hz), sum(1 - W_hz)))
  
  # ── 2. Propensity forest + overlap trimming ─────────────────────────────────
  W_hat_mod <- tryCatch(
    regression_forest(X_hz, W_hz, num.trees = NUM_TREES_NF, seed = SEED),
    error = function(e) { cat("  SKIP: propensity model failed.\n"); NULL }
  )
  if (is.null(W_hat_mod)) next
  
  ps   <- predict(W_hat_mod)$predictions
  keep <- ps > PS_TRIM_LOW & ps < PS_TRIM_HIGH
  
  cat(sprintf("  After overlap trim (%.2f–%.2f): n = %d\n",
              PS_TRIM_LOW, PS_TRIM_HIGH, sum(keep)))
  
  if (sum(keep) < MIN_OBS) {
    cat("  SKIP: too few observations after overlap trim.\n"); next
  }
  
  Xk   <- X_hz[keep, , drop = FALSE]
  Yk   <- Y_hz[keep]
  Wk   <- W_hz[keep]
  idk  <- ids_hz[keep]
  
  # ── 3. Nuisance models on trimmed sample ───────────────────────────────────
  cat("  Fitting nuisance models...\n")
  Y_forest <- regression_forest(Xk, Yk, num.trees = NUM_TREES_NF, seed = SEED)
  W_forest <- regression_forest(Xk, Wk, num.trees = NUM_TREES_NF, seed = SEED)
  Y_hat    <- predict(Y_forest)$predictions
  W_hat    <- predict(W_forest)$predictions
  
  # ── 4. Causal forest ────────────────────────────────────────────────────────
  cat("  Fitting causal forest...\n")
  cf <- causal_forest(
    Xk, Yk, Wk,
    Y.hat      = Y_hat,
    W.hat      = W_hat,
    num.trees  = NUM_TREES_CF,
    seed       = SEED
  )
  
  # ── 5. Average treatment effects ────────────────────────────────────────────
  ate_r <- as.list(average_treatment_effect(cf, target.sample = "all"))
  att_r <- as.list(average_treatment_effect(cf, target.sample = "treated"))
  atc_r <- as.list(average_treatment_effect(cf, target.sample = "control"))
  ato_r <- as.list(average_treatment_effect(cf, target.sample = "overlap"))
  
  cat("\n  ── Average Treatment Effects ──\n")
  cat(sprintf("  ATE : %s\n", fmt_effect(ate_r$estimate, ate_r$std.err)))
  cat(sprintf("  ATT : %s\n", fmt_effect(att_r$estimate, att_r$std.err)))
  cat(sprintf("  ATC : %s\n", fmt_effect(atc_r$estimate, atc_r$std.err)))
  cat(sprintf("  ATO : %s\n", fmt_effect(ato_r$estimate, ato_r$std.err)))
  
  # ── 6. Calibration test (heterogeneity validity) ────────────────────────────
  # Tests whether the forest's CATE predictions are informative.
  # differential.forest.prediction ~ 1 means CATEs track real heterogeneity.
  cat("\n  ── Calibration Test (CATE validity) ──\n")
  cal_res <- test_calibration(cf)
  print(cal_res)
  cal     <- extract_cal(cal_res)
  
  # ── 7. RATE test (can CATEs be used for targeting?) ─────────────────────────
  # RATE = Rank-Weighted Average Treatment Effect.
  # A significant positive RATE means high-CATE units genuinely benefit more —
  # i.e., the CATEs are useful for prioritisation/targeting.
  cat("\n  ── RATE Test (targeting value of CATEs) ──\n")
  rate_res <- tryCatch(
    rank_average_treatment_effect(cf, priorities = predict(cf)$predictions),
    error = function(e) { cat("  RATE failed:", conditionMessage(e), "\n"); NULL }
  )
  
  if (!is.null(rate_res)) {
    print(rate_res)
    rate <- extract_rate(rate_res)
    cat(sprintf("  RATE: %.4f (SE=%.4f, p=%.3f)%s\n",
                rate$est, rate$se, rate$p, p_to_stars(rate$p)))
  } else {
    rate <- list(est = NA_real_, se = NA_real_, p = NA_real_)
  }
  
  # ── 8. Variable importance ──────────────────────────────────────────────────
  # 8a. CATE variable importance (only meaningful if heterogeneity is validated)
  vi_vals <- variable_importance(cf)
  vi_df   <- data.frame(
    hazard     = hz,
    model      = "CATE",
    variable   = varnames,
    importance = as.numeric(vi_vals)
  )
  
  # 8b. Nuisance variable importance (always meaningful: what predicts Y and W)
  vi_Y <- variable_importance(Y_forest)
  vi_W <- variable_importance(W_forest)
  vi_df <- bind_rows(
    vi_df,
    data.frame(hazard = hz, model = "Y_nuisance",
               variable = varnames, importance = as.numeric(vi_Y)),
    data.frame(hazard = hz, model = "W_nuisance",
               variable = varnames, importance = as.numeric(vi_W))
  )
  varimp_all <- bind_rows(varimp_all, vi_df)
  
  top5 <- vi_df |> filter(model == "CATE") |> arrange(desc(importance)) |> head(5)
  cat("\n  Top-5 CATE variable importances:\n")
  for (i in seq_len(nrow(top5))) {
    cat(sprintf("    %d. %-35s %.4f\n", i, top5$variable[i], top5$importance[i]))
  }
  top5Y <- vi_df |> filter(model == "Y_nuisance") |> arrange(desc(importance)) |> head(5)
  cat("  Top-5 outcome (Y) nuisance importances:\n")
  for (i in seq_len(nrow(top5Y))) {
    cat(sprintf("    %d. %-35s %.4f\n", i, top5Y$variable[i], top5Y$importance[i]))
  }
  
  # ── 9. CATEs per observation ────────────────────────────────────────────────
  # Note: estimate.variance requires honesty (default in causal_forest).
  # variance.estimates can be NULL in edge cases, so we guard against it.
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
    treated  = Wk,
    ps       = W_hat
  )
  cates_all <- bind_rows(cates_all, cate_df)
  
  # ── 10. Collate summary row ─────────────────────────────────────────────────
  summary_row <- data.frame(
    hazard          = hz,
    n_total         = sum(valid),
    n_keep          = sum(keep),
    # ATE
    ATE_est         = ate_r$estimate,
    ATE_se          = ate_r$std.err,
    ATE_p           = 2 * (1 - pnorm(abs(ate_r$estimate / ate_r$std.err))),
    ATE_fmt         = fmt_effect(ate_r$estimate, ate_r$std.err),
    # ATT
    ATT_est         = att_r$estimate,
    ATT_se          = att_r$std.err,
    ATT_p           = 2 * (1 - pnorm(abs(att_r$estimate / att_r$std.err))),
    ATT_fmt         = fmt_effect(att_r$estimate, att_r$std.err),
    # ATC
    ATC_est         = atc_r$estimate,
    ATC_se          = atc_r$std.err,
    ATC_p           = 2 * (1 - pnorm(abs(atc_r$estimate / atc_r$std.err))),
    ATC_fmt         = fmt_effect(atc_r$estimate, atc_r$std.err),
    # ATO
    ATO_est         = ato_r$estimate,
    ATO_se          = ato_r$std.err,
    ATO_p           = 2 * (1 - pnorm(abs(ato_r$estimate / ato_r$std.err))),
    ATO_fmt         = fmt_effect(ato_r$estimate, ato_r$std.err),
    # Calibration
    cal_est         = cal$est,
    cal_se          = cal$se,
    cal_p           = cal$p,
    cal_fmt         = fmt_effect(cal$est, cal$se),
    # RATE
    rate_est        = rate$est,
    rate_se         = rate$se,
    rate_p          = rate$p,
    rate_fmt        = fmt_effect(rate$est, rate$se),
    stringsAsFactors = FALSE
  )
  summary_all <- bind_rows(summary_all, summary_row)
  
}  # end hazard loop

# ------------------------------------------------------------------------------
# SAVE OUTPUTS
# ------------------------------------------------------------------------------

cat("\n\nSaving results...\n")
write.csv(summary_all, file.path(OUTPUT_DIR, "cf_summary.csv"),  row.names = FALSE)
write.csv(cates_all,   file.path(OUTPUT_DIR, "cf_cates.csv"),    row.names = FALSE)
write.csv(varimp_all,  file.path(OUTPUT_DIR, "cf_varimp.csv"),   row.names = FALSE)

# ------------------------------------------------------------------------------
# SUBGROUP ANALYSIS
# Theoretically motivated median splits on key moderators.
# This is valid regardless of CATE validity — we're estimating ATEs within
# subgroups defined a priori, not using the forest's CATE predictions.
# ------------------------------------------------------------------------------

cat("\n\n", paste(rep("=", 80), collapse = ""), "\n")
cat("SUBGROUP ANALYSIS (theoretically motivated median splits)\n")
cat(paste(rep("=", 80), collapse = ""), "\n\n")

# Moderator variables and readable labels
moderators <- list(
  nat_amenity_score    = "Natural Amenities",
  inequality_ratio     = "Inequality",
  POPULATION           = "Population",
  pop_density          = "Population Density",
  gdp_pc               = "GDP per Capita",
  emp_100k             = "Number of Employed",
  education_index      = "Education Index",
  income_price_ratio2023 = "Price-Income"
)

subgroup_all <- data.frame()

for (hz in hazards) {
  
  W <- tryCatch({
    med <- median(df[[hz]], na.rm = TRUE)
    if (!is.finite(med)) stop("non-finite median")
    ifelse(df[[hz]] >= med, 1L, 0L)
  }, error = function(e) {
    cat("  SKIP: could not compute median —", conditionMessage(e), "\n"); NULL
  })
  if (is.null(W)) next
  
  valid <- !is.na(W) & !is.na(Y)
  X_hz  <- X.cf[valid, , drop = FALSE]
  Y_hz  <- Y[valid]
  W_hz  <- W[valid]
  df_hz <- df[valid, ]          # keep original df columns for moderator splits
  
  if (nrow(X_hz) < MIN_OBS || length(unique(W_hz)) < 2) next
  
  # Propensity trim (same as main loop)
  W_hat_mod <- tryCatch(
    regression_forest(X_hz, W_hz, num.trees = NUM_TREES_NF, seed = SEED),
    error = function(e) NULL
  )
  if (is.null(W_hat_mod)) next
  ps   <- predict(W_hat_mod)$predictions
  keep <- ps > PS_TRIM_LOW & ps < PS_TRIM_HIGH
  if (sum(keep) < MIN_OBS) next
  
  Xk   <- X_hz[keep, , drop = FALSE]
  Yk   <- Y_hz[keep]
  Wk   <- W_hz[keep]
  df_k <- df_hz[keep, ]
  
  Y_hat <- predict(regression_forest(Xk, Yk, num.trees = NUM_TREES_NF, seed = SEED))$predictions
  W_hat <- predict(regression_forest(Xk, Wk, num.trees = NUM_TREES_NF, seed = SEED))$predictions
  
  cf <- causal_forest(
    Xk, Yk, Wk,
    Y.hat     = Y_hat,
    W.hat     = W_hat,
    num.trees = NUM_TREES_CF,
    seed      = SEED
  )
  
  cat(sprintf("\nHAZARD: %s\n", hz))
  
  for (mod_var in names(moderators)) {
    mod_label <- moderators[[mod_var]]
    
    if (!mod_var %in% colnames(df_k)) next
    mod_vals <- df_k[[mod_var]]
    if (all(is.na(mod_vals))) next
    
    # ── Split definitions: median and 25/75 percentile ──────────────────────
    splits <- list(
      median = list(
        label    = "median",
        high_idx = !is.na(mod_vals) & mod_vals >= median(mod_vals, na.rm = TRUE),
        low_idx  = !is.na(mod_vals) & mod_vals <  median(mod_vals, na.rm = TRUE)
      ),
      pct2575 = list(
        label    = "p25/p75",
        high_idx = !is.na(mod_vals) & mod_vals >= quantile(mod_vals, 0.75, na.rm = TRUE),
        low_idx  = !is.na(mod_vals) & mod_vals <= quantile(mod_vals, 0.25, na.rm = TRUE)
      )
    )
    
    for (sp in splits) {
      
      high_idx <- sp$high_idx
      low_idx  <- sp$low_idx
      
      # Need at least MIN_OBS in each group and treatment variation
      if (sum(high_idx) < MIN_OBS || sum(low_idx) < MIN_OBS) next
      if (length(unique(Wk[high_idx])) < 2 || length(unique(Wk[low_idx])) < 2) next
      
      ate_high <- tryCatch(
        as.list(average_treatment_effect(cf, subset = high_idx)),
        error = function(e) list(estimate = NA_real_, std.err = NA_real_)
      )
      ate_low <- tryCatch(
        as.list(average_treatment_effect(cf, subset = low_idx)),
        error = function(e) list(estimate = NA_real_, std.err = NA_real_)
      )
      
      # Interaction test: difference in ATEs between subgroups
      diff_est <- ate_high$estimate - ate_low$estimate
      diff_se  <- sqrt(ate_high$std.err^2 + ate_low$std.err^2)
      diff_p   <- 2 * (1 - pnorm(abs(diff_est / diff_se)))
      
      cat(sprintf(
        "  %-22s [%-8s] | High(n=%d): %s  |  Low(n=%d): %s  |  Diff: %.4f (p=%.3f)%s\n",
        mod_label, sp$label,
        sum(high_idx), fmt_effect(ate_high$estimate, ate_high$std.err),
        sum(low_idx),  fmt_effect(ate_low$estimate,  ate_low$std.err),
        diff_est, diff_p, p_to_stars(diff_p)
      ))
      
      subgroup_all <- bind_rows(subgroup_all, data.frame(
        hazard        = hz,
        moderator     = mod_label,
        mod_var       = mod_var,
        split_type    = sp$label,
        n_high        = sum(high_idx),
        n_low         = sum(low_idx),
        ATE_high_est  = ate_high$estimate,
        ATE_high_se   = ate_high$std.err,
        ATE_high_p    = 2 * (1 - pnorm(abs(ate_high$estimate / ate_high$std.err))),
        ATE_high_fmt  = fmt_effect(ate_high$estimate, ate_high$std.err),
        ATE_low_est   = ate_low$estimate,
        ATE_low_se    = ate_low$std.err,
        ATE_low_p     = 2 * (1 - pnorm(abs(ate_low$estimate / ate_low$std.err))),
        ATE_low_fmt   = fmt_effect(ate_low$estimate, ate_low$std.err),
        diff_est      = diff_est,
        diff_se       = diff_se,
        diff_p        = diff_p,
        diff_fmt      = fmt_effect(diff_est, diff_se),
        stringsAsFactors = FALSE
      ))
    }
  }
}

write.csv(subgroup_all, file.path(OUTPUT_DIR, "cf_subgroups.csv"), row.names = FALSE)
cat("\nSubgroup results saved to cf_subgroups.csv\n")

# ------------------------------------------------------------------------------
# PRETTY SUMMARY TABLE (console)
# ------------------------------------------------------------------------------

cat("\n", paste(rep("=", 80), collapse = ""), "\n")
cat("SUMMARY TABLE\n")
cat(paste(rep("=", 80), collapse = ""), "\n\n")

display_cols <- c("hazard", "n_keep", "ATE_fmt", "ATT_fmt", "ATC_fmt", "ATO_fmt",
                  "cal_fmt", "rate_fmt")
display_names <- c("Hazard", "N", "ATE", "ATT", "ATC", "ATO",
                   "Calibration (het.)", "RATE (targeting)")

if (nrow(summary_all) > 0) {
  out <- summary_all[, display_cols]
  colnames(out) <- display_names
  print(out, row.names = FALSE)
}

cat("\n\nNote on interpretation:\n")
cat("  ATE  = average effect over full overlap-trimmed sample\n")
cat("  ATT  = average effect for the treated (high-risk) group\n")
cat("  ATC  = average effect for the control (low-risk) group\n")
cat("  ATO  = overlap-weighted ATE (most robust to PS model misspecification)\n")
cat("  Calibration = test_calibration() differential.forest.prediction;\n")
cat("               significant result (p<0.05) means CATEs are statistically\n")
cat("               meaningful and not just noise.\n")
cat("  RATE = rank_average_treatment_effect(); significant positive result means\n")
cat("         CATEs are useful for targeting — high-CATE counties truly benefit more.\n")
cat("\nSignificance codes: *** p<0.001  ** p<0.01  * p<0.05  . p<0.1\n")
