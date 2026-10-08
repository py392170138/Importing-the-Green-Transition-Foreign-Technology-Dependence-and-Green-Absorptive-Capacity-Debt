source(analysis_project_file("03_代码", "R", "00_utils.R"))
source(analysis_project_file("03_代码", "R", "22_shift_share_audit.R"))

test_that("scaled shock contributions exactly reconstruct clipped instruments", {
  d <- shift_share_fixture()
  pieces <- decompose_main_instruments(d$sample, d$shocks)
  rebuilt <- aggregate(
    cbind(z_piece, z_gad_piece) ~ economy_id + treatment_time,
    pieces,
    sum
  )
  rebuilt <- rebuilt[order(rebuilt$economy_id, rebuilt$treatment_time), ]
  sample <- d$sample[order(d$sample$economy_id, d$sample$treatment_time), ]
  expect_equal(rebuilt$z_piece, sample$z_a, tolerance = 1e-12)
  expect_equal(rebuilt$z_gad_piece, sample$z_gad_a, tolerance = 1e-12)
})

test_that("generalized Rotemberg weights sum to one and report signs", {
  audit <- generalized_rotemberg(shift_share_fixture())
  expect_equal(
    sum(audit$weights$signed_weight), 1.0, tolerance = 1e-10
  )
  expect_equal(
    audit$summary$absolute_weight_sum,
    sum(abs(audit$weights$signed_weight)),
    tolerance = 1e-10
  )
  expect_gte(audit$summary$hhi_absolute, 0)
  expect_lte(audit$summary$top5_absolute_share, 1.0 + 1e-12)
  expect_equal(audit$summary$shock_clusters, 2L)
})

test_that("nonzero clipped instruments cannot be scaled from a zero raw value", {
  d <- shift_share_fixture()
  d$sample$raw_Z[[1L]] <- 0
  expect_error(
    decompose_main_instruments(d$sample, d$shocks),
    "cannot reconstruct clipped Z from zero raw Z"
  )
})

test_that("unavailable shock covariance preserves generalized weights", {
  d <- shift_share_fixture()
  d$shocks$exporter <- "X"
  d$shocks$hs6 <- "000001"
  audit <- generalized_rotemberg(d)
  expect_equal(audit$summary$shock_inference_status, "unavailable")
  expect_equal(sum(audit$weights$signed_weight), 1, tolerance = 1e-10)
  expect_true(is.na(audit$summary$shock_std_error_gimc))
  expect_true(is.na(audit$summary$shock_std_error_interaction))
})

test_that("cross-moment inversion respects the registered numerical rank", {
  q <- diag(c(1, 3e-16))
  expect_true(all(is.finite(solve(q))))
  ranked <- ranked_cross_moment_inverse(q)
  expect_equal(ranked$rank, 1L)
  expect_null(ranked$inverse)
})
