analysis_project_file <- function(...) {
  root <- Sys.getenv("GREEN_DEBT_PROJECT_ROOT", unset = NA_character_)
  if (is.na(root) || !nzchar(root)) {
    stop("GREEN_DEBT_PROJECT_ROOT is not set by the R test harness")
  }
  file.path(root, ...)
}

synthetic_cell <- function(
  outcome_id = "synthetic", horizon = 3L,
  gad_version = "gad_core", sample_version = "core_complete_case"
) {
  list(
    outcome_id = outcome_id, horizon = horizon,
    gad_version = gad_version, sample_version = sample_version,
    role = "synthetic", analysis_family = "confirmatory"
  )
}

synthetic_run_context <- function() {
  list(
    run_id = "synthetic-run", spec_id = "gad_lp_iv_v1",
    input_authority_hash = paste(rep("a", 64L), collapse = ""),
    git_commit = paste(rep("b", 40L), collapse = ""),
    renv_lock_sha256 = paste(rep("c", 64L), collapse = ""),
    evidence_policy_sha256 = paste(rep("d", 64L), collapse = ""),
    seed = 20260820L, created_at_utc = "2026-08-29T00:00:00Z"
  )
}

run_context_fixture <- synthetic_run_context

synthetic_bundle <- function(beta, theta, covariance) {
  dimnames(covariance) <- list(
    c("gimc_a", "gimc_gad_a"),
    c("gimc_a", "gimc_gad_a")
  )
  list(
    coefficients = c(gimc_a = beta, gimc_gad_a = theta),
    covariance = covariance,
    reference_df = 59
  )
}

threshold_spec_fixture <- function(
  outcome_id = "green_industrial_upgrading_index"
) {
  list(
    outcome_id = outcome_id, horizon = 5L,
    gad_version = "gad_no_supp",
    sample_version = "core_complete_case",
    percentiles = 20:80, minimum_regime_share = 0.20,
    criterion = "minimum_two_way_fe_ssr",
    quantile_type = 7L,
    tie_break = "lowest_percentile_then_lowest_q",
    seed = 20260820L
  )
}

threshold_fixture <- function(n_economies = 30L, years = 10L) {
  n <- n_economies * years
  economy <- rep(sprintf("E%03d", seq_len(n_economies)), each = years)
  treatment_time <- rep(2000L + seq_len(years) - 1L, n_economies)
  gad <- seq(0, 2, length.out = n)
  z <- sin(seq_len(n))
  gimc <- 0.8 * z + cos(seq_len(n)) / 5
  data.frame(
    economy_id = economy, treatment_time = treatment_time,
    delta_outcome = 1.2 * gimc * (gad <= 1) - 0.3 * gimc * (gad > 1),
    gimc_a = gimc, gimc_gad_a = gimc * gad, gad_a = gad,
    z_a = z, z_gad_a = z * gad,
    renewable_energy_consumption_share_a = 0,
    trade_openness_a = 0,
    industry_value_added_share_a = 0,
    gdp_per_capita_a = 0
  )
}

synthetic_iv_fixture <- function(first_stage) {
  d <- threshold_fixture(40L, 12L)
  d$gimc_a <- first_stage * d$z_a + cos(seq_len(nrow(d)))
  d$gimc_gad_a <- d$gimc_a * d$gad_a
  d$delta_outcome <- 1.5 * d$gimc_a - 0.4 * d$gimc_gad_a +
    sin(seq_len(nrow(d))) / 3
  d
}

synthetic_iv_fit <- function(d) {
  fit_lp_iv(d, synthetic_cell(), synthetic_run_context())
}

inference_spec <- function() {
  list(
    ar_grid_points_per_axis = 121L,
    ar_initial_se_span = 6,
    ar_max_expansions = 4L,
    alpha = 0.05
  )
}

shift_share_fixture <- function() {
  sample <- data.frame(
    economy_id = c("A", "B"), treatment_time = c(2001L, 2001L),
    raw_Z = c(0.3, 0.1), raw_Z_GAD = c(0.15, 0.1),
    raw_gad_lag = c(0.5, 1.0), z_a = c(0.3, 0.1),
    z_gad_a = c(0.15, 0.1), gimc_a = c(0.4, 0.2),
    gimc_gad_a = c(0.2, 0.2), residual = c(0.1, -0.2)
  )
  shocks <- data.frame(
    economy_id = c("A", "A", "B", "B"),
    treatment_time = rep(2001L, 4L),
    exporter = c("X", "Y", "X", "Y"),
    hs6 = c("000001", "000002", "000001", "000002"),
    year = rep(2001L, 4L),
    contribution = c(0.1, 0.2, 0.04, 0.06)
  )
  list(sample = sample, shocks = shocks)
}

reporting_fixture <- function() {
  estimates <- data.frame(
    outcome_id = rep(
      c("co2_tonnes_per_million_current_usd", "green_export_complexity"),
      each = 2L
    ),
    horizon = rep(c(1L, 2L), 2L), term = "gimc_a",
    estimate = c(-0.2, -0.3, 0.1, 0.2),
    std_error = rep(0.05, 4L)
  )
  list(estimates = estimates)
}
