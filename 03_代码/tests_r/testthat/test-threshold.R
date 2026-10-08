source(analysis_project_file("03_代码", "R", "00_utils.R"))
source(analysis_project_file("03_代码", "R", "10_lp_models.R"))
source(analysis_project_file("03_代码", "R", "11_inference.R"))
source(analysis_project_file("03_代码", "R", "20_threshold.R"))

test_that("threshold selection uses the literal grid, shares, criterion, and tie break", {
  d <- threshold_fixture(n_economies = 30L, years = 10L)
  result <- choose_threshold(d, threshold_spec_fixture())
  expect_gte(result$percentile, 20L)
  expect_lte(result$percentile, 80L)
  expect_gte(result$low_share, 0.20)
  expect_gte(result$high_share, 0.20)
  expect_equal(result$criterion, "minimum_two_way_fe_ssr")
  ordered <- result$candidates[order(
    result$candidates$ssr,
    result$candidates$percentile,
    result$candidates$q
  ), ]
  expect_equal(result$q, ordered$q[[1L]])
})

test_that("an existing registry can only be verified, never reselected", {
  path <- tempfile(fileext = ".json")
  first <- select_and_freeze_threshold(
    threshold_fixture(), path, threshold_spec_fixture(), run_context_fixture(),
    bootstrap_draws = 19L, minimum_valid_draws = 10L
  )
  second <- select_and_freeze_threshold(
    threshold_fixture(), path, threshold_spec_fixture(), run_context_fixture(),
    bootstrap_draws = 19L, minimum_valid_draws = 10L
  )
  expect_identical(first$registry_hash, second$registry_hash)
  changed <- threshold_spec_fixture(outcome_id = "green_export_complexity")
  expect_error(
    select_and_freeze_threshold(
      threshold_fixture(), path, changed, run_context_fixture(),
      bootstrap_draws = 19L, minimum_valid_draws = 10L
    ),
    "frozen threshold registry mismatch"
  )
})

test_that("bootstrap relabels resampled economy blocks", {
  d <- threshold_fixture(n_economies = 8L, years = 5L)
  resampled <- resample_economy_blocks(d, rep("E001", 3L))
  expect_equal(length(unique(resampled$economy_id)), 3L)
  expect_equal(nrow(resampled), 15L)
  expect_equal(
    anyDuplicated(paste(resampled$economy_id, resampled$treatment_time)),
    0L
  )
})

test_that("threshold IV rows bind both regimes to the frozen registry", {
  d <- synthetic_iv_fixture(first_stage = 1.0)
  registry <- list(
    selection_outcome = "green_industrial_upgrading_index",
    registry_hash = paste(rep("d", 64L), collapse = ""),
    sample_hash = paste(rep("e", 64L), collapse = ""),
    q = 1.0
  )
  bundle <- fit_threshold_iv(
    d, synthetic_cell(), synthetic_run_context(), registry$q
  )
  rows <- threshold_estimate_rows(bundle, d, registry)
  expect_true(all(c(
    "difference_estimate", "difference_std_error",
    "difference_conf_low", "difference_conf_high", "difference_p_value",
    "difference_wild_estimate", "difference_wild_std_error",
    "difference_wild_conf_low", "difference_wild_conf_high",
    "difference_wild_p_value", "difference_wild_inference_status",
    "difference_wild_draws", "difference_wild_seed"
  ) %in% names(rows)))
  expect_setequal(rows$regime, c("low", "high"))
  expect_true(all(rows$q == registry$q))
  expect_true(all(rows$registry_hash == registry$registry_hash))
  expect_true(all(rows$registry_sample_hash == registry$sample_hash))
  expect_true(all(rows$selection_outcome == registry$selection_outcome))
  expect_equal(sum(rows$regime_n), nrow(d))
  expect_true(all(rows$difference_estimate ==
    bundle$coefficients[["gimc_high"]] - bundle$coefficients[["gimc_low"]]))
  expected_variance <- bundle$covariance["gimc_high", "gimc_high"] +
    bundle$covariance["gimc_low", "gimc_low"] -
    2 * bundle$covariance["gimc_high", "gimc_low"]
  expect_true(all(rows$difference_std_error == sqrt(expected_variance)))
  expect_equal(rows$difference_estimate[[1L]], rows$difference_estimate[[2L]])
})

test_that("20-29 cluster threshold inference freezes all three wild tests", {
  d <- threshold_fixture(n_economies = 25L, years = 12L)
  d$gimc_a <- d$z_a + cos(seq_len(nrow(d))) / 4
  d$gimc_gad_a <- d$gimc_a * d$gad_a
  bundle <- fit_threshold_iv(
    d, synthetic_cell(), synthetic_run_context(), q = 1.0
  )
  planner <- get("threshold_inference_plan", mode = "function")
  plan <- planner(bundle, seed = 20260820L, draws = 9999L)
  expect_equal(plan$status, "wild_bootstrap_required")
  expect_equal(plan$parameters, c("gimc_low", "gimc_high", "high_minus_low"))
  expect_equal(plan$seed, 20260820L)
  expect_equal(plan$draws, 9999L)
})

test_that("threshold difference reparameterization is same-sample algebraic equivalence", {
  d <- synthetic_iv_fixture(first_stage = 1.0)
  bundle <- fit_threshold_iv(
    d, synthetic_cell(), synthetic_run_context(), q = 1.0
  )
  fitter <- get("fit_threshold_difference_iv", mode = "function")
  difference <- fitter(d, synthetic_cell(), q = 1.0)
  expect_equal(stats::nobs(difference$fit), bundle$n)
  expect_equal(
    unname(stats::coef(difference$fit)[["gimc_high"]]),
    unname(bundle$coefficients[["gimc_high"]] -
      bundle$coefficients[["gimc_low"]]),
    tolerance = 1e-8
  )
})
