library(testthat)
library(arrow)
source(analysis_project_file("03_代码", "R", "00_utils.R"))
source(analysis_project_file("03_代码", "R", "10_lp_models.R"))

test_that("runner model result identity has no empty list argument", {
  expressions <- parse(
    file = analysis_project_file("03_代码", "R", "run_analysis.R")
  )
  is_definition <- vapply(expressions, function(expression) {
    is.call(expression) && identical(expression[[1L]], quote(`<-`)) &&
      identical(expression[[2L]], quote(model_row_base))
  }, logical(1L))
  expect_equal(sum(is_definition), 1L)
  eval(expressions[[which(is_definition)]], envir = environment())
  row <- model_row_base(
    synthetic_run_context(), "lp_fe", synthetic_cell(),
    data.frame(treatment_time = c(2000L, 2022L)), 30L
  )
  expect_equal(row$year_min, 2000L)
  expect_equal(row$year_max, 2022L)
  expect_equal(row$reference_distribution, "cluster_t")
})

test_that("LP-IV recovers both structural coefficients", {
  d <- as.data.frame(read_parquet(
    analysis_project_file(
      "03_代码", "tests", "fixtures", "analysis",
      "synthetic_lp_panel.parquet"
    )
  ))
  bundle <- fit_lp_iv(d, synthetic_cell(), synthetic_run_context())
  expect_lt(abs(bundle$coefficients[["gimc_a"]] - 2.0), 0.30)
  expect_lt(abs(bundle$coefficients[["gimc_gad_a"]] + 0.5), 0.30)
  expect_equal(bundle$fixed_effects, c("economy_id", "treatment_time"))
  expect_equal(bundle$cluster, "economy_id")
})

test_that("LP-FE uses the exact same fixed effects and cluster", {
  d <- as.data.frame(read_parquet(
    analysis_project_file(
      "03_代码", "tests", "fixtures", "analysis",
      "synthetic_lp_panel.parquet"
    )
  ))
  bundle <- fit_lp_fe(d, synthetic_cell(), synthetic_run_context())
  expect_equal(bundle$fixed_effects, c("economy_id", "treatment_time"))
  expect_equal(bundle$cluster, "economy_id")
  expect_equal(bundle$n, 1200L)
  expect_equal(bundle$economies, 60L)
  expect_equal(bundle$reference_distribution, "cluster_t")
  expect_equal(bundle$reference_df, 59)
  expect_equal(
    unname(bundle$conf_low[["gimc_a"]]),
    unname(bundle$coefficients[["gimc_a"]]) -
      stats::qt(0.975, 59) * unname(bundle$std_error[["gimc_a"]])
  )
  expect_equal(
    unname(bundle$p_value[["gimc_a"]]),
    2 * stats::pt(
      -abs(unname(bundle$coefficients[["gimc_a"]]) /
        unname(bundle$std_error[["gimc_a"]])),
      df = 59
    )
  )
  expect_equal(bundle$ssc_config, "fixest:K.adj=TRUE;K.fixef=nonnested;K.exact=FALSE;G.adj=TRUE;G.df=min;t.df=min")
})

test_that("model preparation rejects h zero and duplicate economy-year keys", {
  d <- as.data.frame(read_parquet(
    analysis_project_file(
      "03_代码", "tests", "fixtures", "analysis",
      "synthetic_lp_panel.parquet"
    )
  ))
  bad_h <- synthetic_cell(horizon = 0L)
  expect_error(prepare_model_frame(d, bad_h), "horizon zero")
  expect_error(
    prepare_model_frame(rbind(d, d[1L, ]), synthetic_cell()),
    "duplicate economy-time"
  )
})

test_that("marginal effects use the full beta-theta covariance", {
  bundle <- synthetic_bundle(
    beta = 2.0, theta = -0.5,
    covariance = matrix(c(0.04, 0.01, 0.01, 0.09), 2, 2)
  )
  out <- compute_marginal_effects(bundle, gad_values = c(0, 1))
  expect_equal(out$estimate, c(2.0, 1.5))
  expect_equal(out$std_error[[2]], sqrt(0.04 + 0.09 + 2 * 0.01))
  expect_equal(
    out$conf_low[[1L]],
    2.0 - stats::qt(0.975, 59) * sqrt(0.04)
  )
})
