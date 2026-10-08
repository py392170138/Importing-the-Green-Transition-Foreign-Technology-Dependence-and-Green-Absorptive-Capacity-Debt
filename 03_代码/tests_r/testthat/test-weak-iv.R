source(analysis_project_file("03_代码", "R", "00_utils.R"))
source(analysis_project_file("03_代码", "R", "10_lp_models.R"))
source(analysis_project_file("03_代码", "R", "11_inference.R"))
source(analysis_project_file("03_代码", "R", "21_weak_iv.R"))

test_that("AR region classifies bounded and weak grids", {
  strong <- synthetic_iv_fixture(first_stage = 1.0)
  weak <- synthetic_iv_fixture(first_stage = 0.01)
  bounded <- compute_ar_region(
    strong, synthetic_iv_fit(strong), synthetic_cell(), inference_spec()
  )
  weak_result <- compute_ar_region(
    weak, synthetic_iv_fit(weak), synthetic_cell(), inference_spec()
  )
  expect_equal(bounded$status, "bounded")
  expect_true(weak_result$status %in% c("unbounded", "disjoint"))
  expect_equal(bounded$grid_points_per_axis, 121L)
  expect_true(bounded$conventional_point_accepted)
  expect_true(weak_result$conventional_point_accepted)
})

test_that("direct grid membership rejects a projected-only match", {
  accepted <- data.frame(
    beta = c(-1, 1), theta = c(-1, 1),
    beta_index = c(1L, 3L), theta_index = c(1L, 3L),
    p_value = c(0.2, 0.2)
  )
  expect_false(conventional_grid_point_accepted(accepted, points = 3L))
  accepted <- rbind(
    accepted,
    data.frame(
      beta = 0, theta = 0, beta_index = 2L, theta_index = 2L,
      p_value = 0.9
    )
  )
  expect_true(conventional_grid_point_accepted(accepted, points = 3L))
})

test_that("AR evaluation is invariant to row order", {
  d <- synthetic_iv_fixture(first_stage = 0.4)
  set.seed(20260820)
  shuffled <- d[sample.int(nrow(d)), ]
  left <- compute_ar_region(
    d, synthetic_iv_fit(d), synthetic_cell(), inference_spec()
  )
  right <- compute_ar_region(
    shuffled, synthetic_iv_fit(shuffled), synthetic_cell(), inference_spec()
  )
  expect_equal(left$accepted_hash, right$accepted_hash)
})

test_that("vectorized AR p-values equal direct full-design CR2 HTZ", {
  d <- synthetic_iv_fixture(first_stage = 0.4)
  index <- seq_len(nrow(d))
  d$renewable_energy_consumption_share_a <- sin(index * 0.17)
  d$trade_openness_a <- cos(index * 0.11)
  d$industry_value_added_share_a <- sin(index * 0.07 + 0.3)
  d$gdp_per_capita_a <- cos(index * 0.13 + 0.2)
  prepared <- prepare_model_frame(d, synthetic_cell())
  beta <- 1.1
  theta <- -0.2
  fast <- evaluate_ar_grid(
    prepare_ar_evaluator(prepared), beta, theta
  )$p_value[[1L]]
  auxiliary <- prepared
  auxiliary$u_b <- auxiliary$delta_outcome -
    beta * auxiliary$gimc_a - theta * auxiliary$gimc_gad_a
  direct_fit <- stats::lm(
    u_b ~ z_a + z_gad_a + gad_a +
      renewable_energy_consumption_share_a + trade_openness_a +
      industry_value_added_share_a + gdp_per_capita_a +
      factor(economy_id) + factor(treatment_time),
    data = auxiliary
  )
  direct_vcov <- clubSandwich::vcovCR(
    direct_fit, cluster = auxiliary$economy_id, type = "CR2"
  )
  coefficients <- stats::na.omit(stats::coef(direct_fit))
  constraints <- matrix(0, nrow = 2L, ncol = length(coefficients))
  constraints[cbind(1:2, match(c("z_a", "z_gad_a"), names(coefficients)))] <- 1
  direct <- clubSandwich::Wald_test(
    direct_fit, constraints = constraints, vcov = direct_vcov, test = "HTZ"
  )$p_val[[1L]]
  expect_equal(fast, direct, tolerance = 1e-12)
})

test_that("one-axis unbounded projection preserves the other three bounds", {
  accepted <- data.frame(
    beta = c(-3, -2, -1), theta = c(0.2, 0.3, 0.4),
    beta_index = c(1L, 2L, 3L), theta_index = c(2L, 3L, 4L),
    p_value = c(0.1, 0.2, 0.1)
  )
  projected <- project_ar_bounds(accepted, points = 5L)
  expect_true(is.na(projected$beta_low))
  expect_equal(projected$beta_high, -1)
  expect_equal(projected$theta_low, 0.2)
  expect_equal(projected$theta_high, 0.4)
})
