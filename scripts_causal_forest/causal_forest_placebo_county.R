# ==============================================================================
# PLACEBO / PERMUTATION TEST FOR CAUSAL FOREST
# COUNTY-LEVEL VARIANT (exact copy of causal_forest_placebo.R). Differences:
# DATA_PATH/OUTPUT_DIR point at the county table/output folder; clusters =
# state FIPS passed to every forest (see loop_causal_forest_county.R).
# ==============================================================================
# Builds a null distribution for the ATE by randomly permuting treatment
# labels (W) many times and re-running the full causal forest pipeline.
#
# Logic:
#   Under the sharp null (no treatment effect), permuting W destroys any
#   real signal while preserving the covariate structure and sample size.
#   If our real ATE lies in the tail of the permutation distribution, it
#   is unlikely to be a statistical artefact.
#
# Outputs an empirical p-value:
#   p_perm = share of permuted |ATE| >= |real ATE|
#
# Saves: cf_summary_random.csv  (one row per hazard × permutation)
#        cf_placebo_summary.csv  (one row per hazard: real ATE + empirical p)
# ==============================================================================

library(grf)
library(dplyr)

# Cap parallel threads -- grf defaults to num.threads=NULL (all detected
# cores), which is too disruptive on this shared machine. Passed explicitly
# as num.threads = N_THREADS to every regression_forest()/causal_forest() call below.
N_THREADS <- 12

# ------------------------------------------------------------------------------
# CONFIGURATION  — match exactly with main script
# ------------------------------------------------------------------------------

DATA_PATH    <- "data_generated/df_merged_county.csv"
OUTPUT_DIR   <- "data_generated/cf_results/county"
NUM_TREES_CF <- 2000
NUM_TREES_NF <- 500
PS_TRIM_LOW  <- 0.05
PS_TRIM_HIGH <- 0.95
MIN_OBS      <- 50
SEED         <- 42

# Number of permutations. 200 is sufficient to get a stable empirical p-value
# down to 0.005.
N_PERM <- 200

# Restricted to RISK_SCORE only for the permutation test.
hazards_placebo <- c("RISK_SCORE")

set.seed(SEED)

# ------------------------------------------------------------------------------
# DATA  (identical setup to main script)
# ------------------------------------------------------------------------------

df <- read.csv(DATA_PATH)
df[df == -9999] <- NA

Y <- df$pagerank_logdiff

# Cluster by state: every county nests in exactly one state (see loop_causal_forest_county.R).
clusters_all <- as.integer(as.factor(substr(df$STCOFIPS, 1, 2)))

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

# ------------------------------------------------------------------------------
# HELPER: run one causal forest and return ATE given a (possibly permuted) W
# This is a stripped-down version of the main loop — no CATE saving,
# no variable importance, no subgroups. Just the ATE.
# ------------------------------------------------------------------------------

run_cf_ate <- function(W_input, X_full, Y_full, clusters_full, perm_seed) {

  valid       <- !is.na(W_input) & !is.na(Y_full) & complete.cases(X_full)
  X_hz        <- X_full[valid, , drop = FALSE]
  Y_hz        <- Y_full[valid]
  W_hz        <- W_input[valid]
  clusters_hz <- clusters_full[valid]

  if (sum(valid) < MIN_OBS || length(unique(W_hz)) < 2) return(NA_real_)

  # Propensity forest + overlap trim
  W_hat_mod <- tryCatch(
    regression_forest(X_hz, W_hz, num.trees = NUM_TREES_NF, seed = perm_seed,
                       clusters = clusters_hz, num.threads = N_THREADS),
    error = function(e) NULL
  )
  if (is.null(W_hat_mod)) return(NA_real_)

  ps   <- predict(W_hat_mod)$predictions
  keep <- ps > PS_TRIM_LOW & ps < PS_TRIM_HIGH
  if (sum(keep) < MIN_OBS) return(NA_real_)

  Xk         <- X_hz[keep, , drop = FALSE]
  Yk         <- Y_hz[keep]
  Wk         <- W_hz[keep]
  clusters_k <- clusters_hz[keep]

  Y_hat <- tryCatch(
    predict(regression_forest(Xk, Yk, num.trees = NUM_TREES_NF, seed = perm_seed,
                               clusters = clusters_k, num.threads = N_THREADS))$predictions,
    error = function(e) NULL
  )
  W_hat <- tryCatch(
    predict(regression_forest(Xk, Wk, num.trees = NUM_TREES_NF, seed = perm_seed,
                               clusters = clusters_k, num.threads = N_THREADS))$predictions,
    error = function(e) NULL
  )
  if (is.null(Y_hat) || is.null(W_hat)) return(NA_real_)

  cf <- tryCatch(
    causal_forest(Xk, Yk, Wk,
                  Y.hat = Y_hat, W.hat = W_hat,
                  num.trees = NUM_TREES_CF, seed = perm_seed,
                  clusters = clusters_k, num.threads = N_THREADS),
    error = function(e) NULL
  )
  if (is.null(cf)) return(NA_real_)

  ate <- tryCatch(
    as.list(average_treatment_effect(cf, target.sample = "all")),
    error = function(e) list(estimate = NA_real_, std.err = NA_real_)
  )
  c(estimate = ate$estimate, std.err = ate$std.err)
}

# ------------------------------------------------------------------------------
# REAL ATEs  (re-run with same setup to get ground truth for comparison)
# ------------------------------------------------------------------------------

cat("Computing real ATEs...\n")

real_ates <- data.frame()

for (hz in hazards_placebo) {
  cat(sprintf("  %s\n", hz))

  W_real <- tryCatch({
    med <- median(df[[hz]], na.rm = TRUE)
    ifelse(df[[hz]] >= med, 1L, 0L)
  }, error = function(e) NULL)
  if (is.null(W_real)) next

  res <- run_cf_ate(W_real, X.cf, Y, clusters_all, perm_seed = SEED)
  real_ates <- bind_rows(real_ates, data.frame(
    hazard   = hz,
    real_ate = res["estimate"],
    real_se  = res["std.err"]
  ))
}

# ------------------------------------------------------------------------------
# PERMUTATION LOOP
# ------------------------------------------------------------------------------

cat(sprintf("\nRunning %d permutations for %d hazards...\n",
            N_PERM, length(hazards_placebo)))
cat("(This will take a while — approx. 3 causal forests per permutation per hazard)\n\n")

perm_results <- data.frame()

for (hz in hazards_placebo) {

  sep <- paste(rep("=", 60), collapse = "")
  cat(sprintf("\n%s\nHAZARD: %s\n%s\n", sep, hz, sep))

  # Get the real W vector (needed so we permute the right variable)
  W_real <- tryCatch({
    med <- median(df[[hz]], na.rm = TRUE)
    ifelse(df[[hz]] >= med, 1L, 0L)
  }, error = function(e) NULL)
  if (is.null(W_real)) next

  perm_ates <- numeric(N_PERM)

  for (i in seq_len(N_PERM)) {

    perm_seed_i <- SEED + i   # reproducible but different each permutation

    # Permute W — destroys treatment-outcome relationship while preserving
    # covariate structure, sample size, and treatment prevalence
    valid_idx <- which(!is.na(W_real))
    W_perm    <- W_real
    W_perm[valid_idx] <- sample(W_real[valid_idx])

    res_i <- run_cf_ate(W_perm, X.cf, Y, clusters_all, perm_seed = perm_seed_i)
    perm_ates[i] <- res_i["estimate"]

    if (i %% 25 == 0) {
      cat(sprintf("  Permutation %d/%d  |  permuted ATE so far: mean=%.4f, sd=%.4f\n",
                  i, N_PERM,
                  mean(perm_ates[1:i], na.rm = TRUE),
                  sd(perm_ates[1:i],   na.rm = TRUE)))
    }
  }

  perm_results <- bind_rows(perm_results, data.frame(
    hazard   = hz,
    perm_id  = seq_len(N_PERM),
    perm_ate = perm_ates
  ))

  cat(sprintf("  Done. Permuted ATE: mean=%.4f, sd=%.4f\n",
              mean(perm_ates, na.rm = TRUE), sd(perm_ates, na.rm = TRUE)))
}

# ------------------------------------------------------------------------------
# EMPIRICAL P-VALUES AND SUMMARY
# ------------------------------------------------------------------------------

cat("\n\nComputing empirical p-values...\n")

placebo_summary <- perm_results |>
  group_by(hazard) |>
  summarise(
    n_perm        = sum(!is.na(perm_ate)),
    perm_mean     = mean(perm_ate, na.rm = TRUE),
    perm_sd       = sd(perm_ate,   na.rm = TRUE),
    perm_p025     = quantile(perm_ate, 0.025, na.rm = TRUE),
    perm_p975     = quantile(perm_ate, 0.975, na.rm = TRUE),
    .groups       = "drop"
  ) |>
  left_join(real_ates, by = "hazard") |>
  left_join(
    # Empirical p-value: share of permutations with |ATE| >= |real ATE|
    perm_results |>
      left_join(real_ates, by = "hazard") |>
      group_by(hazard) |>
      summarise(
        p_perm = mean(abs(perm_ate) >= abs(real_ate), na.rm = TRUE),
        .groups = "drop"
      ),
    by = "hazard"
  )

# ------------------------------------------------------------------------------
# PRINT SUMMARY
# ------------------------------------------------------------------------------

cat("\n", paste(rep("=", 80), collapse = ""), "\n")
cat("PLACEBO TEST SUMMARY\n")
cat(paste(rep("=", 80), collapse = ""), "\n\n")

for (i in seq_len(nrow(placebo_summary))) {
  r <- placebo_summary[i, ]
  cat(sprintf(
    "%-15s | Real ATE: %7.4f (SE=%.4f) | Perm: mean=%7.4f, sd=%.4f, 95%% CI [%.4f, %.4f] | p_perm=%.3f%s\n",
    r$hazard, r$real_ate, r$real_se,
    r$perm_mean, r$perm_sd, r$perm_p025, r$perm_p975,
    r$p_perm,
    ifelse(r$p_perm < 0.001, "***",
    ifelse(r$p_perm < 0.01,  "**",
    ifelse(r$p_perm < 0.05,  "*",
    ifelse(r$p_perm < 0.1,   ".", ""))))
  ))
}

cat("\n\nInterpretation:\n")
cat("  p_perm = share of permuted |ATE| values >= |real ATE|.\n")
cat("  A small p_perm means the real ATE is unlikely under the sharp null.\n")
cat("  Permuted ATEs should be centred near zero — if not, there is a bug.\n")
cat("  The permuted SD gives you the noise floor: real ATEs >> perm SD are robust.\n")

# ------------------------------------------------------------------------------
# SAVE
# ------------------------------------------------------------------------------

write.csv(perm_results,    file.path(OUTPUT_DIR, "cf_summary_random.csv"),  row.names = FALSE)
write.csv(placebo_summary, file.path(OUTPUT_DIR, "cf_placebo_summary.csv"), row.names = FALSE)
cat("\nSaved cf_summary_random.csv and cf_placebo_summary.csv\n")
