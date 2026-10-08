library(testthat)
source(analysis_project_file("03_代码", "R", "00_utils.R"))
source(analysis_project_file("03_代码", "R", "10_lp_models.R"))
source(analysis_project_file("03_代码", "R", "11_inference.R"))

test_that("cluster inference thresholds are literal", {
  expect_equal(classify_inference(19L, 2L), "exploratory_lt20_clusters")
  expect_equal(classify_inference(20L, 2L), "wild_bootstrap_required")
  expect_equal(classify_inference(29L, 2L), "wild_bootstrap_required")
  expect_equal(classify_inference(30L, 2L), "cluster_robust")
  expect_equal(classify_inference(60L, 1L), "fail_rank_deficient")
})

test_that("R first-stage diagnostics are deterministic and full rank", {
  d <- as.data.frame(arrow::read_parquet(
    analysis_project_file(
      "03_代码", "tests", "fixtures", "analysis",
      "synthetic_lp_panel.parquet"
    )
  ))
  left <- compute_first_stage_metrics(d)
  right <- compute_first_stage_metrics(d[nrow(d):1L, ])
  expect_equal(left$rank, 2L)
  expect_equal(left, right, tolerance = 1e-10)
  expect_true(all(is.finite(unlist(left))))
})

test_that("R first-stage nuisance design partials out GAD", {
  d <- as.data.frame(arrow::read_parquet(
    analysis_project_file(
      "03_代码", "tests", "fixtures", "analysis",
      "synthetic_lp_panel.parquet"
    )
  ))
  index <- seq_len(nrow(d))
  d$gad_a <- sin(index^2 * 0.017)
  residualized_gad <- qr_residualize(
    as.matrix(d["gad_a"]), fixed_effect_design(d)
  )
  expect_lt(max(abs(residualized_gad)), 1e-12)
})

test_that("atomic CSV preserves exact finite-double identities", {
  value <- 1.4777372886633888
  path <- tempfile(fileext = ".csv")
  on.exit(unlink(path), add = TRUE)
  atomic_write_csv(data.frame(q = value), path)
  round_tripped <- utils::read.csv(path)
  expect_identical(round_tripped$q[[1L]], value)
})

test_that("frozen model package identity matches the evidence-policy authority", {
  versions <- jsonlite::fromJSON(FROZEN_PACKAGE_VERSIONS_JSON)
  policy <- jsonlite::read_json(
    analysis_project_file("config", "evidence_policy.json"),
    simplifyVector = TRUE
  )
  keys <- sort(names(versions))
  expect_setequal(keys, names(policy$software$packages))
  expect_identical(versions[keys], policy$software$packages[keys])
})

test_that("Julia IV bootstrap runtime is frozen when installed", {
  skip_if(!nzchar(Sys.which("julia")), "Julia is not installed")
  runtime <- validate_julia_runtime(Sys.getenv("GREEN_DEBT_PROJECT_ROOT"))
  expect_equal(runtime$julia_version, "1.12.7")
  expect_equal(runtime$wild_boot_tests_version, "0.9.8")
})

test_that("locked Julia backend returns an IV wild-bootstrap interval", {
  skip_if(!nzchar(Sys.which("julia")), "Julia is not installed")
  d <- as.data.frame(arrow::read_parquet(
    analysis_project_file(
      "03_代码", "tests", "fixtures", "analysis",
      "synthetic_lp_panel.parquet"
    )
  ))
  d <- d[d$economy_id %in% sprintf("E%03d", 0:19), , drop = FALSE]
  result <- extract_wild_result(
    suppressWarnings(
      run_wild_iv(d, "gimc_a", seed = 20260820L, draws = 199L)
    )
  )
  expect_equal(result$draws, 199L)
  expect_true(is.finite(result$conf_low))
  expect_true(is.finite(result$conf_high))
  expect_lte(result$conf_low, result$conf_high)
  JuliaConnectoR::stopJulia()
})

test_that("unbounded wild-bootstrap intervals remain explicit", {
  raw <- structure(
    list(
      conf_int = matrix(c(-Inf, Inf), nrow = 1L),
      p_val = 0.4,
      boot_iter = 9999L,
      point_estimate = 0.2,
      t_stat = 1
    ),
    class = "boottest"
  )
  result <- extract_wild_result(raw)
  expect_equal(result$interval_status, "unbounded")
  expect_true(is.na(result$conf_low))
  expect_true(is.na(result$conf_high))
  expect_equal(result$p_value, 0.4)
  expect_equal(result$draws, 9999L)
  expect_equal(result$estimate, 0.2)
  expect_equal(result$std_error, 0.2)
})
