source(analysis_project_file("03_代码", "R", "30_reporting.R"))

reporting_test_data_root <- function() {
  root <- normalizePath(analysis_project_file(), mustWork = TRUE)
  if (basename(dirname(root)) == ".worktrees") dirname(dirname(root)) else root
}

threshold_branch_fixture <- function() {
  horizons <- rep(1:5, each = 2L)
  regimes <- rep(c("low", "high"), 5L)
  clusters <- rep(c(19L, 25L, 25L, 25L, 30L), each = 2L)
  inference_status <- rep(c(
    "exploratory_lt20_clusters", "wild_bootstrap_required",
    "wild_bootstrap_unbounded", "wild_bootstrap_unbounded",
    "cluster_robust"
  ), each = 2L)
  wild_status <- c(
    "not_required", "not_required",
    "wild_bootstrap_bounded", "wild_bootstrap_bounded",
    "wild_bootstrap_unbounded", "wild_bootstrap_unbounded",
    "wild_bootstrap_bounded", "wild_bootstrap_unbounded",
    "not_required", "not_required"
  )
  difference_wild_status <- rep(c(
    "not_required", "wild_bootstrap_bounded",
    "wild_bootstrap_unbounded", "wild_bootstrap_bounded",
    "not_required"
  ), each = 2L)
  conventional_low <- -c(101, 102, 201, 202, 301, 302, 401, 402, 501, 502)
  conventional_high <- -conventional_low
  conventional_p <- c(
    .101, .102, .201, .202, .301, .302, .401, .402, .501, .502
  )
  wild_low <- c(NA, NA, -21, -22, NA, NA, -41, NA, NA, NA)
  wild_high <- c(NA, NA, 21, 22, NA, NA, 41, NA, NA, NA)
  wild_p <- c(NA, NA, .021, .022, .031, .032, .041, .042, NA, NA)
  difference_low <- rep(-c(103, 203, 303, 403, 503), each = 2L)
  difference_high <- -difference_low
  difference_p <- rep(c(.103, .203, .303, .403, .503), each = 2L)
  difference_wild_low <- rep(c(NA, -23, NA, -43, NA), each = 2L)
  difference_wild_high <- rep(c(NA, 23, NA, 43, NA), each = 2L)
  difference_wild_p <- rep(c(NA, .023, .033, .043, NA), each = 2L)

  data.frame(
    run_id = "run-threshold-matrix", spec_id = "threshold-v1",
    input_authority_hash = "authority-hash", git_commit = "fixture-commit",
    renv_lock_sha256 = "renv-hash", evidence_policy_sha256 = "policy-hash",
    created_at_utc = "2026-08-30T00:00:00Z", year_min = 2000L,
    year_max = 2022L, r_version = "4.6.1", python_version = "3.12.13",
    julia_version = NA_character_, package_versions_json = "{}",
    random_seed = 20260820L, ssc_config = "CR2",
    reference_distribution = "cluster_t", reference_df = clusters - 1L,
    estimator = "lp_fe", analysis_family = "threshold",
    selection_outcome = "green_industrial_upgrade_index",
    registry_hash = "registry-hash", registry_sample_hash = "sample-hash",
    q = 1.4777372886633888, outcome_id = "green_export_complexity",
    horizon = horizons, gad_version = "gad_no_supp",
    sample_version = "core_complete_case", regime = regimes,
    estimate = c(1.01, -1.02, 2.01, -2.02, 3.01, -3.02, 4.01, -4.02, 5.01, -5.02),
    std_error = c(.11, .12, .21, .22, .31, .32, .41, .42, .51, .52),
    conf_low = conventional_low, conf_high = conventional_high,
    p_value = conventional_p, wild_estimate = c(
      1.01, -1.02, 2.01, -2.02, 3.01, -3.02, 4.01, -4.02, 5.01, -5.02
    ),
    wild_std_error = c(.11, .12, .21, .22, .31, .32, .41, .42, .51, .52),
    wild_conf_low = wild_low, wild_conf_high = wild_high,
    wild_p_value = wild_p, wild_inference_status = wild_status,
    wild_draws = ifelse(clusters >= 20L & clusters < 30L, 9999L, NA_integer_),
    wild_seed = ifelse(clusters >= 20L & clusters < 30L, 20260820L, NA_integer_),
    low_high_covariance = rep(c(.013, .023, .033, .043, .053), each = 2L),
    difference_estimate = rep(c(-1.03, -2.03, -3.03, -4.03, -5.03), each = 2L),
    difference_std_error = rep(c(.13, .23, .33, .43, .53), each = 2L),
    difference_conf_low = difference_low,
    difference_conf_high = difference_high,
    difference_p_value = difference_p,
    difference_wild_estimate = rep(c(-1.03, -2.03, -3.03, -4.03, -5.03), each = 2L),
    difference_wild_std_error = rep(c(.13, .23, .33, .43, .53), each = 2L),
    difference_wild_conf_low = difference_wild_low,
    difference_wild_conf_high = difference_wild_high,
    difference_wild_p_value = difference_wild_p,
    difference_wild_inference_status = difference_wild_status,
    difference_wild_draws = ifelse(clusters >= 20L & clusters < 30L, 9999L, NA_integer_),
    difference_wild_seed = ifelse(
      clusters >= 20L & clusters < 30L, 20260820L, NA_integer_
    ),
    regime_n = rep(c(19L, 25L, 25L, 25L, 30L), each = 2L),
    regime_share = .5, n = clusters * 2L, economies = clusters,
    clusters = clusters, first_stage_status = "not_applicable",
    inference_status = inference_status,
    stringsAsFactors = FALSE
  )
}

test_that("production runner exports project root for reporting identity", {
  runner <- readLines(
    analysis_project_file("03_代码", "R", "run_analysis.R"),
    warn = FALSE
  )
  expect_true(any(grepl(
    "GREEN_DEBT_PROJECT_ROOT\\s*=\\s*project_root", runner
  )))
})

test_that("table cells are direct joins from machine estimates", {
  inputs <- reporting_fixture()
  rendered <- build_main_tables(inputs)
  expected <- inputs$estimates[
    inputs$estimates$term == "gimc_a",
    c("outcome_id", "horizon", "estimate", "std_error")
  ]
  expect_equal(
    rendered$lp_iv[, names(expected)],
    expected[order(expected$outcome_id, expected$horizon), ]
  )
})

test_that("figure data and plotted data have the same canonical hash", {
  inputs <- reporting_fixture()
  figure <- build_dynamic_path_figure(inputs)
  expect_equal(
    canonical_reporting_data_hash(figure$data),
    attr(figure$plot, "source_data_hash")
  )
})

test_that("figure source hashes ignore sub-tolerance float noise only", {
  formal <- data.frame(id = c("a", "b"), value = c(1, -2))
  bounded <- formal
  bounded$value <- bounded$value + c(5e-14, -5e-14)
  changed <- formal
  changed$value[[1L]] <- changed$value[[1L]] + 2e-12

  expect_equal(
    canonical_reporting_data_hash(formal),
    canonical_reporting_data_hash(bounded)
  )
  expect_false(identical(
    canonical_reporting_data_hash(formal),
    canonical_reporting_data_hash(changed)
  ))
})

test_that("figure hash tolerance is strict across signs and scales", {
  values <- data.frame(
    id = letters[1:5],
    value = c(-1e9, -1, 0, 1, 1e9),
    stringsAsFactors = FALSE
  )
  bounded <- values
  bounded$value <- bounded$value + c(-2e-7, -2e-15, 2e-15, 2e-15, 2e-7)
  changed <- values
  changed$value <- changed$value + c(-2e-3, -2e-12, 2e-12, 2e-12, 2e-3)

  expect_equal(
    canonical_reporting_data_hash(values),
    canonical_reporting_data_hash(bounded)
  )
  expect_false(identical(
    canonical_reporting_data_hash(values),
    canonical_reporting_data_hash(changed)
  ))
})

test_that("figure source hashes are invariant to logical row order", {
  formal <- data.frame(
    group = c("b", "a", "c"),
    value = c(2.5, -1.25, 0),
    stringsAsFactors = FALSE
  )

  expect_equal(
    canonical_reporting_data_hash(formal),
    canonical_reporting_data_hash(formal[c(3L, 1L, 2L), , drop = FALSE])
  )
})

test_that("publication intervals follow the registered cluster rule", {
  values <- data.frame(
    clusters = c(19L, 20L, 29L, 30L, 20L),
    inference_status = c(
      "exploratory_lt20_clusters", "wild_bootstrap_required",
      "wild_bootstrap_required", "cluster_robust",
      "wild_bootstrap_unbounded"
    ),
    conf_low = rep(-0.5, 5L), conf_high = rep(0.5, 5L),
    wild_conf_low = c(NA, -0.2, -0.3, NA, NA),
    wild_conf_high = c(NA, 0.2, 0.3, NA, NA)
  )
  selected <- select_publication_intervals(values)
  expect_equal(
    selected$publication_interval_status,
    c(
      "point_only_exploratory", "wild_bootstrap", "wild_bootstrap",
      "cluster_robust", "wild_bootstrap_unbounded"
    )
  )
  expect_equal(
    selected$publication_conf_low,
    c(NA, -0.2, -0.3, -0.5, NA)
  )
  expect_equal(
    selected$publication_conf_high,
    c(NA, 0.2, 0.3, 0.5, NA)
  )
})

test_that("marginal curve never substitutes a normal band for wild inference", {
  expect_equal(
    marginal_curve_interval_policy(19L, "exploratory_lt20_clusters"),
    "point_only_exploratory"
  )
  expect_equal(
    marginal_curve_interval_policy(20L, "wild_bootstrap_required"),
    "point_only_no_registered_wild_curve"
  )
  expect_equal(
    marginal_curve_interval_policy(29L, "wild_bootstrap_unbounded"),
    "point_only_wild_bootstrap_unbounded"
  )
  expect_equal(
    marginal_curve_interval_policy(30L, "cluster_robust"),
    "cluster_robust"
  )
})

test_that("Table 6 consumes only validated high-minus-low fields", {
  threshold <- data.frame(
    outcome_id = c("green_export_complexity", "green_export_complexity"),
    horizon = c(5L, 5L), gad_version = "gad_no_supp",
    sample_version = "core_complete_case", regime = c("low", "high"),
    estimate = c(1, -0.2), conf_low = c(0.7, -0.5),
    conf_high = c(1.3, 0.1), p_value = c(0.01, 0.2),
    wild_conf_low = NA_real_, wild_conf_high = NA_real_,
    wild_p_value = NA_real_, wild_inference_status = "not_required",
    clusters = 30L, inference_status = "cluster_robust",
    difference_estimate = c(9.9, 9.9),
    difference_std_error = c(0.4, 0.4),
    difference_conf_low = c(9.0, 9.0),
    difference_conf_high = c(10.8, 10.8),
    difference_p_value = c(0.03, 0.03),
    difference_wild_conf_low = NA_real_,
    difference_wild_conf_high = NA_real_,
    difference_wild_p_value = NA_real_,
    difference_wild_inference_status = "not_required",
    stringsAsFactors = FALSE
  )
  builder <- get("build_threshold_table", mode = "function")
  rendered <- builder(threshold)
  expect_equal(nrow(rendered), 1L)
  expect_equal(rendered$difference_estimate, 9.9)
  expect_equal(rendered$difference_conf_low, 9.0)
  expect_equal(rendered$difference_conf_high, 10.8)
  expect_equal(rendered$difference_p_value, 0.03)
  expect_equal(rendered$registered_interval_source, "conventional_cluster_t")
})

test_that("Table 6 never falls back to conventional low or high intervals when wild is unbounded", {
  threshold <- data.frame(
    run_id = "run-threshold-20", spec_id = "threshold-v1",
    input_authority_hash = "authority-hash", git_commit = "a083981",
    renv_lock_sha256 = "renv-hash", evidence_policy_sha256 = "policy-hash",
    created_at_utc = "2026-08-30T00:00:00Z", year_min = 2000L,
    year_max = 2022L, r_version = "4.4.0", python_version = "3.12.0",
    julia_version = NA_character_, package_versions_json = "{}",
    random_seed = 20260820L, ssc_config = "CR2", reference_distribution = "t",
    reference_df = 19, estimator = "lp_fe", analysis_family = "threshold",
    selection_outcome = "green_industrial_upgrade_index",
    registry_hash = "registry-hash", registry_sample_hash = "sample-hash",
    q = 1.4777372886633888,
    outcome_id = "green_export_complexity", horizon = 5L,
    gad_version = "gad_no_supp", sample_version = "core_complete_case",
    regime = c("low", "high"), estimate = c(1.5, -0.5),
    std_error = c(0.2, 0.3), conf_low = c(-9, -7),
    conf_high = c(6, 8), p_value = c(0.01, 0.02),
    wild_estimate = c(1.5, -0.5), wild_std_error = c(0.2, 0.3),
    wild_conf_low = NA_real_, wild_conf_high = NA_real_,
    wild_p_value = c(0.04, 0.05),
    wild_inference_status = "wild_bootstrap_unbounded",
    wild_draws = 9999L, wild_seed = 20260820L,
    low_high_covariance = c(0.03, 0.03),
    difference_estimate = c(-2, -2), difference_std_error = c(0.4, 0.4),
    difference_conf_low = c(-10, -10), difference_conf_high = c(9, 9),
    difference_p_value = c(0.03, 0.03),
    difference_wild_estimate = c(-2, -2),
    difference_wild_std_error = c(0.4, 0.4),
    difference_wild_conf_low = NA_real_, difference_wild_conf_high = NA_real_,
    difference_wild_p_value = c(0.06, 0.06),
    difference_wild_inference_status = "wild_bootstrap_unbounded",
    difference_wild_draws = 9999L, difference_wild_seed = 20260820L,
    regime_n = c(20L, 20L), regime_share = c(0.5, 0.5), n = 40L,
    economies = 20L, clusters = 20L, first_stage_status = "not_applicable",
    inference_status = "wild_bootstrap_unbounded",
    stringsAsFactors = FALSE
  )

  rendered <- build_threshold_table(threshold)

  expect_true(is.na(rendered$low_conf_low))
  expect_true(is.na(rendered$low_conf_high))
  expect_true(is.na(rendered$high_conf_low))
  expect_true(is.na(rendered$high_conf_high))
  expect_equal(rendered$low_conventional_conf_low, -9)
  expect_equal(rendered$high_conventional_conf_high, 8)
  expect_equal(
    rendered$registered_low_interval_source,
    "wild_bootstrap_unbounded_point_only"
  )
  expect_equal(
    rendered$registered_high_interval_source,
    "wild_bootstrap_unbounded_point_only"
  )
  expect_equal(
    rendered$registered_interval_source,
    "wild_bootstrap_unbounded_point_only"
  )
  expect_match(rendered$publication_warning, "LOW")
  expect_match(rendered$publication_warning, "HIGH")
  expect_match(rendered$publication_warning, "DIFFERENCE")
})

test_that("Table 6 projects every registered interval branch and mixed state end to end", {
  rendered <- build_threshold_table(threshold_branch_fixture())
  rendered <- rendered[order(rendered$horizon), , drop = FALSE]

  expect_equal(rendered$low_conf_low, c(NA, -21, NA, -41, -501))
  expect_equal(rendered$low_conf_high, c(NA, 21, NA, 41, 501))
  expect_equal(rendered$low_p_value, c(NA, .021, .031, .041, .501))
  expect_equal(rendered$high_conf_low, c(NA, -22, NA, NA, -502))
  expect_equal(rendered$high_conf_high, c(NA, 22, NA, NA, 502))
  expect_equal(rendered$high_p_value, c(NA, .022, .032, .042, .502))
  expect_equal(
    rendered$registered_difference_conf_low,
    c(NA, -23, NA, -43, -503)
  )
  expect_equal(
    rendered$registered_difference_conf_high,
    c(NA, 23, NA, 43, 503)
  )
  expect_equal(
    rendered$registered_difference_p_value,
    c(NA, .023, .033, .043, .503)
  )

  expect_equal(
    rendered$registered_low_interval_source,
    c(
      "point_only_exploratory_no_registered_interval",
      "wild_cluster_bootstrap_9999",
      "wild_bootstrap_unbounded_point_only",
      "wild_cluster_bootstrap_9999",
      "conventional_cluster_t"
    )
  )
  expect_equal(
    rendered$registered_high_interval_source,
    c(
      "point_only_exploratory_no_registered_interval",
      "wild_cluster_bootstrap_9999",
      "wild_bootstrap_unbounded_point_only",
      "wild_bootstrap_unbounded_point_only",
      "conventional_cluster_t"
    )
  )
  expect_equal(
    rendered$registered_interval_source,
    c(
      "point_only_exploratory_no_registered_interval",
      "wild_cluster_bootstrap_9999",
      "wild_bootstrap_unbounded_point_only",
      "wild_cluster_bootstrap_9999",
      "conventional_cluster_t"
    )
  )

  expect_equal(rendered$low_conventional_conf_low, -c(101, 201, 301, 401, 501))
  expect_equal(rendered$low_conventional_conf_high, c(101, 201, 301, 401, 501))
  expect_equal(rendered$low_conventional_p_value, c(.101, .201, .301, .401, .501))
  expect_equal(rendered$high_conventional_conf_low, -c(102, 202, 302, 402, 502))
  expect_equal(rendered$high_conventional_conf_high, c(102, 202, 302, 402, 502))
  expect_equal(rendered$high_conventional_p_value, c(.102, .202, .302, .402, .502))
  expect_equal(rendered$difference_conf_low, -c(103, 203, 303, 403, 503))
  expect_equal(rendered$difference_conf_high, c(103, 203, 303, 403, 503))
  expect_equal(rendered$difference_p_value, c(.103, .203, .303, .403, .503))

  expect_equal(
    rendered$publication_warning,
    c(
      paste(
        "LOW: POINT-ONLY/EXPLORATORY",
        "HIGH: POINT-ONLY/EXPLORATORY",
        "DIFFERENCE: POINT-ONLY/EXPLORATORY",
        sep = " | "
      ),
      paste(
        "LOW: LOW-CLUSTER WILD BOOTSTRAP",
        "HIGH: LOW-CLUSTER WILD BOOTSTRAP",
        "DIFFERENCE: LOW-CLUSTER WILD BOOTSTRAP",
        sep = " | "
      ),
      paste(
        "LOW: POINT-ONLY/UNBOUNDED WILD INTERVAL",
        "HIGH: POINT-ONLY/UNBOUNDED WILD INTERVAL",
        "DIFFERENCE: POINT-ONLY/UNBOUNDED WILD INTERVAL",
        sep = " | "
      ),
      paste(
        "LOW: LOW-CLUSTER WILD BOOTSTRAP",
        "HIGH: POINT-ONLY/UNBOUNDED WILD INTERVAL",
        "DIFFERENCE: LOW-CLUSTER WILD BOOTSTRAP",
        sep = " | "
      ),
      paste(
        "LOW: CONVENTIONAL CLUSTER-t",
        "HIGH: CONVENTIONAL CLUSTER-t",
        "DIFFERENCE: CONVENTIONAL CLUSTER-t",
        sep = " | "
      )
    )
  )
})

test_that("reporting warnings visibly distinguish point-only and unbounded cells", {
  formatter <- get("publication_warning_text", mode = "function")
  values <- data.frame(
    clusters = c(18L, 25L, 25L, 40L),
    inference_status = c(
      "exploratory_lt20_clusters", "wild_bootstrap_required",
      "wild_bootstrap_unbounded", "cluster_robust"
    )
  )
  warning <- formatter(values)
  expect_match(warning, "POINT-ONLY/EXPLORATORY")
  expect_match(warning, "LOW-CLUSTER WILD BOOTSTRAP")
  expect_match(warning, "UNBOUNDED")
})

test_that("industrial publication surfaces preserve the frozen exploratory evidence gate", {
  inputs <- load_reporting_inputs(analysis_project_file("06_结果", "analysis"))
  expect_identical(
    as.character(inputs$evidence_policy$concentration$status),
    "unresolved_no_preregistered_cutoff"
  )
  expect_identical(
    as.character(inputs$evidence_policy$concentration$unresolved_policy),
    "exploratory"
  )
  expected_warning <- paste0(
    "EVIDENCE GATE: EXPLORATORY — concentration=",
    "unresolved_no_preregistered_cutoff"
  )

  table_5 <- publication_tables(inputs)[[5L]]
  expect_true(all(c(
    "concentration_status", "evidence_grade", "publication_warning"
  ) %in% names(table_5)))
  expect_true(all(
    table_5$concentration_status == "unresolved_no_preregistered_cutoff"
  ))
  expect_true(all(table_5$evidence_grade == "exploratory"))
  expect_true(all(grepl(
    expected_warning, table_5$publication_warning, fixed = TRUE
  )))

  figures <- publication_figures(inputs)
  for (index in c(4L, 5L)) {
    data <- figures[[index]]$data
    expect_true(all(c(
      "concentration_status", "evidence_grade", "publication_warning"
    ) %in% names(data)))
    expect_true(all(
      data$concentration_status == "unresolved_no_preregistered_cutoff"
    ))
    expect_true(all(data$evidence_grade == "exploratory"))
    expect_true(all(grepl(expected_warning, data$publication_warning, fixed = TRUE)))
    visible_labels <- paste(unlist(
      figures[[index]]$plot$labels[c("title", "subtitle", "caption")],
      use.names = FALSE
    ), collapse = " | ")
    expect_true(grepl("Exploratory evidence", visible_labels, fixed = TRUE))
    expect_false(grepl(expected_warning, visible_labels, fixed = TRUE))
    expect_true(grepl(
      "concentration cutoff was not preregistered",
      figures[[index]]$metadata$caption,
      fixed = TRUE
    ))
  }
})

test_that("publication figures carry complete SSCI figure contracts without omitting evidence", {
  inputs <- load_reporting_inputs(analysis_project_file("06_结果", "analysis"))
  figures <- publication_figures(inputs)

  expect_length(figures, 7L)
  metadata <- lapply(figures, `[[`, "metadata")
  required <- c(
    "figure_id", "research_question", "role", "panels", "analysis_unit",
    "uncertainty_display", "target_canvas_mm", "first_citation", "caption",
    "output_formats", "minimum_text_pt", "source_files"
  )
  has_contracts <- all(vapply(
    metadata,
    function(item) identical(sort(names(item)), sort(required)),
    logical(1L)
  ))
  expect_true(has_contracts)
  if (!has_contracts) return(invisible())
  expect_equal(
    vapply(metadata, `[[`, character(1L), "figure_id"),
    sprintf("figure_%d", seq_len(7L))
  )
  expect_true(all(vapply(metadata, `[[`, character(1L), "role") == "main"))
  expect_equal(
    vapply(metadata, function(item) item$target_canvas_mm$width, numeric(1L)),
    rep(183, 7L)
  )
  expect_equal(
    vapply(metadata, function(item) item$target_canvas_mm$height, numeric(1L)),
    c(160, 150, 105, 175, 105, 105, 180)
  )
  expect_true(all(vapply(
    metadata,
    function(item) identical(item$output_formats, c("svg", "pdf", "png")),
    logical(1L)
  )))
  expect_true(all(vapply(metadata, `[[`, numeric(1L), "minimum_text_pt") >= 6.5))
  expect_true(all(nzchar(vapply(metadata, `[[`, character(1L), "caption"))))
  expect_identical(metadata[[1L]]$first_citation, "not_available_no_manuscript")

  expect_equal(nrow(figures[[1L]]$data), nrow(inputs$distributions))
  expect_equal(nrow(figures[[2L]]$data), nrow(inputs$descriptive_paths))
  expect_equal(nrow(figures[[4L]]$data), 120L)
  expect_equal(nrow(figures[[7L]]$data), 28L)
})

test_that("publication plots use reader-facing small multiples and preserve registered uncertainty", {
  inputs <- load_reporting_inputs(analysis_project_file("06_结果", "analysis"))
  figures <- publication_figures(inputs)

  panel_counts <- vapply(figures, function(figure) {
    nrow(ggplot2::ggplot_build(figure$plot)$layout$layout)
  }, integer(1L))
  expect_equal(panel_counts, c(8L, 8L, 3L, 5L, 1L, 2L, 2L))

  display_columns <- list(
    c("metric_label", "specification_label"),
    c("outcome_label", "domain_label"),
    c("outcome_label", "interval_label"),
    c("outcome_label", "series_label"),
    character(),
    c("outcome_label", "interval_label"),
    c("panel_label", "display_label")
  )
  for (index in seq_along(figures)) {
    expect_true(all(display_columns[[index]] %in% names(figures[[index]]$data)))
    for (column in display_columns[[index]]) {
      expect_false(any(grepl("_", figures[[index]]$data[[column]], fixed = TRUE)))
    }
    labels <- paste(
      unlist(figures[[index]]$plot$labels, use.names = FALSE),
      collapse = " | "
    )
    expect_false(grepl("_", labels, fixed = TRUE))
  }

  expect_setequal(
    unique(figures[[3L]]$data$interval_label),
    c("Registered 95% interval", "Point only — unbounded")
  )
  expect_setequal(
    unique(figures[[4L]]$data$series_label),
    c("Interaction", "GAD Q25", "GAD Q50", "GAD Q75")
  )
  expect_true(all(is.finite(figures[[4L]]$data$publication_conf_low)))
  expect_true(all(is.finite(figures[[4L]]$data$publication_conf_high)))
  expect_setequal(
    unique(figures[[6L]]$data$interval_label),
    c("Registered 95% interval", "Point only — low clusters")
  )

  wrapped_facet_labels <- c(
    as.character(figures[[1L]]$data$metric_label),
    as.character(figures[[2L]]$data$outcome_label)
  )
  line_widths <- nchar(unlist(strsplit(wrapped_facet_labels, "\n", fixed = TRUE)))
  expect_lte(max(line_widths), 24L)
  displayed_panel_labels <- c(
    as.character(figures[[1L]]$data$panel_label),
    as.character(figures[[2L]]$data$panel_label)
  )
  displayed_line_widths <- nchar(unlist(strsplit(
    displayed_panel_labels, "\n", fixed = TRUE
  )))
  expect_lte(max(displayed_line_widths), 22L)
  expect_length(
    ggplot2::ggplot_build(figures[[7L]]$plot)$layout$panel_scales_x,
    2L
  )
})

test_that("publication theme renders a restrained editorial hierarchy", {
  theme <- publication_theme()

  expect_s3_class(theme$strip.background, "element_blank")
  expect_s3_class(theme$panel.grid.major.x, "element_blank")
  expect_s3_class(theme$panel.grid.minor, "element_blank")
  expect_identical(theme$strip.text$hjust, 0)
  expect_gte(theme$strip.text$size, 8.5)
  expect_gte(theme$axis.text$size, 6.5)
})

test_that("interval-heavy figures use horizontal dot-interval grammar without trend lines", {
  inputs <- load_reporting_inputs(analysis_project_file("06_结果", "analysis"))
  figures <- publication_figures(inputs)

  for (index in c(3L, 4L, 6L)) {
    plot <- figures[[index]]$plot
    expect_identical(rlang::as_label(plot$mapping$x), "estimate")
    expect_false(any(vapply(
      plot$layers,
      function(layer) inherits(layer$geom, "GeomLine"),
      logical(1L)
    )))
    expect_true(any(vapply(
      plot$layers,
      function(layer) identical(layer$geom_params$orientation, "y"),
      logical(1L)
    )))
  }
})

test_that("interval-heavy figures abbreviate large axis values without scientific notation", {
  inputs <- load_reporting_inputs(analysis_project_file("06_结果", "analysis"))
  figures <- publication_figures(inputs)

  for (index in c(3L, 4L, 6L)) {
    scale <- figures[[index]]$plot$scales$get_scales("x")
    expect_false(is.null(scale))
    if (is.null(scale)) next
    expect_equal(scale$labels(c(-4e5, 0, 4e5)), c("-400K", "0", "400K"))
  }
})

test_that("descriptive and diagnostic figures directly label terminal values", {
  inputs <- load_reporting_inputs(analysis_project_file("06_结果", "analysis"))
  figures <- publication_figures(inputs)

  has_text_layer <- function(plot) {
    any(vapply(
      plot$layers,
      function(layer) inherits(layer$geom, "GeomText"),
      logical(1L)
    ))
  }
  expect_true(has_text_layer(figures[[2L]]$plot))
  expect_true(has_text_layer(figures[[7L]]$plot))

  endpoints <- figures[[2L]]$data
  endpoints <- endpoints[!is.na(endpoints$endpoint_label), , drop = FALSE]
  observed <- stats::setNames(endpoints$endpoint_label, endpoints$outcome_id)
  expect_equal(unname(observed[c(
    "co2_tonnes_per_million_current_usd",
    "renewable_capacity_additions_mw_per_million",
    "energy_intensity_mj_per_ppp_gdp",
    "future_green_rca_entry_rate",
    "green_export_complexity",
    "green_export_share",
    "domestic_value_added_share",
    "foreign_value_added_dependence"
  )]), c(
    "-113", "10.4", "-0.253", "0.0891",
    "0.086", "0.00631", "-0.758", "0.0216"
  ))
})

test_that("direct figure annotations remain at or above the SSCI minimum size", {
  inputs <- load_reporting_inputs(analysis_project_file("06_结果", "analysis"))
  figures <- publication_figures(inputs)
  millimetres_per_point <- 25.4 / 72.27
  minimum_size_mm <- 6.5 * millimetres_per_point

  for (index in c(2L, 5L, 7L)) {
    text_layers <- Filter(
      function(layer) inherits(layer$geom, "GeomText"),
      figures[[index]]$plot$layers
    )
    expect_true(length(text_layers) > 0L)
    sizes <- vapply(
      text_layers,
      function(layer) as.numeric(layer$aes_params$size),
      numeric(1L)
    )
    expect_gte(min(sizes), minimum_size_mm)
  }
})

test_that("publication figure construction emits no deprecated-geometry warning", {
  inputs <- load_reporting_inputs(analysis_project_file("06_结果", "analysis"))
  expect_warning(publication_figures(inputs), NA)
})

test_that("the fixed publication set carries data and provenance companions", {
  source_root <- analysis_project_file("06_结果", "analysis")
  inputs <- load_reporting_inputs(source_root)
  output_root <- tempfile("reporting-output-")
  dir.create(output_root)
  result <- render_reporting_outputs(inputs, output_root)

  expect_length(result$tables, 8L)
  expect_length(result$figures, 7L)
  expect_true(all(file.exists(result$tables)))
  has_all_formats <- all(vapply(
    result$figures,
    function(paths) {
      identical(sort(names(paths)), c("pdf", "png", "svg")) &&
        all(file.exists(paths))
    },
    logical(1L)
  ))
  expect_true(has_all_formats)
  if (!has_all_formats) return(invisible())
  png_chunk_types <- function(path) {
    payload <- readBin(path, what = "raw", n = file.info(path)$size)
    expect_identical(
      payload[seq_len(8L)],
      as.raw(c(137L, 80L, 78L, 71L, 13L, 10L, 26L, 10L))
    )
    offset <- 9L
    result <- character()
    while (offset <= length(payload)) {
      size_bytes <- as.integer(payload[offset + 0:3])
      chunk_size <- sum(size_bytes * 256^(3:0))
      chunk_type <- rawToChar(payload[offset + 4:7])
      result <- c(result, chunk_type)
      offset <- offset + 12L + chunk_size
      if (identical(chunk_type, "IEND")) break
    }
    result
  }
  expect_true(all(vapply(
    result$figures,
    function(paths) "sRGB" %in% png_chunk_types(paths[["png"]]),
    logical(1L)
  )))
  expect_true(file.exists(result$manifest))
  expect_true(file.exists(result$captions))
  figure_ids <- sprintf("figure_%d", seq_len(7L))
  expect_true(all(file.exists(file.path(
    output_root, "figures", "data", paste0(figure_ids, ".parquet")
  ))))
  provenance_paths <- file.path(
    output_root, "figures", paste0(figure_ids, ".provenance.json")
  )
  expect_true(all(file.exists(provenance_paths)))
  provenance <- lapply(provenance_paths, jsonlite::read_json, simplifyVector = TRUE)
  expect_true(all(vapply(
    provenance,
    function(item) length(item$source_table_hashes) > 0L,
    logical(1L)
  )))
  reporting_commit <- system2(
    "git", c("-C", shQuote(Sys.getenv("GREEN_DEBT_PROJECT_ROOT")), "rev-parse", "HEAD"),
    stdout = TRUE
  )
  expect_true(all(vapply(
    provenance,
    function(item) identical(item$upstream_model_git_commit, inputs$run_context$git_commit),
    logical(1L)
  )))
  expect_true(all(vapply(
    provenance,
    function(item) identical(item$reporting_git_commit, reporting_commit),
    logical(1L)
  )))
  expect_true(all(vapply(seq_along(provenance), function(index) {
    expected <- vapply(
      result$figures[[index]],
      digest::digest,
      character(1L),
      algo = "sha256", serialize = FALSE, file = TRUE
    )
    identical(
      unname(unlist(provenance[[index]]$rendered_files_sha256)),
      unname(expected[names(result$figures[[index]])])
    )
  }, logical(1L))))
  manifest <- jsonlite::read_json(result$manifest, simplifyVector = FALSE)
  expect_identical(manifest$schema_version, "1.0")
  expect_equal(
    vapply(manifest$figures, `[[`, character(1L), "figure_id"),
    figure_ids
  )
  captions <- readLines(result$captions, warn = FALSE, encoding = "UTF-8")
  expect_equal(sum(grepl("^## Figure [1-7]$", captions)), 7L)
  figure_7_data <- as.data.frame(arrow::read_parquet(file.path(
    output_root, "figures", "data", "figure_7.parquet"
  )))
  expect_setequal(
    unique(figure_7_data$section),
    c("top_contributors", "concentration")
  )
})

test_that("publication failure revokes manifest without mixing publication sets", {
  inputs <- load_reporting_inputs(analysis_project_file("06_结果", "analysis"))
  output_root <- tempfile("reporting-transaction-")
  dir.create(file.path(output_root, "tables"), recursive = TRUE)
  dir.create(file.path(output_root, "figures"), recursive = TRUE)
  writeLines("old manifest", file.path(output_root, "run_manifest.json"))
  writeLines("old table", file.path(output_root, "tables", "old.csv"))
  writeLines("old figure", file.path(output_root, "figures", "old.pdf"))

  expect_error(
    publish_reporting_transaction(inputs, output_root, fail_after_artifact = 2L),
    "injected reporting failure"
  )
  expect_false(file.exists(file.path(output_root, "run_manifest.json")))
  expect_equal(
    list.files(file.path(output_root, "tables")),
    "old.csv"
  )
  expect_equal(
    list.files(file.path(output_root, "figures")),
    "old.pdf"
  )
})

test_that("reporting entry revokes success before loading inputs", {
  output_root <- tempfile("reporting-invalid-input-")
  dir.create(output_root)
  manifest <- file.path(output_root, "run_manifest.json")
  writeLines("old manifest", manifest)
  expect_error(run_reporting(output_root), "validated reporting input is missing")
  expect_false(file.exists(manifest))
})

test_that("reporting rejects an unauthorized root before revoking success", {
  output_root <- tempfile(
    "reporting-unauthorized-", tmpdir = analysis_project_file("06_结果")
  )
  dir.create(output_root)
  on.exit(if (dir.exists(output_root)) unlink(output_root, recursive = TRUE), add = TRUE)
  manifest <- file.path(output_root, "run_manifest.json")
  writeLines("prior success", manifest)

  expect_error(
    run_reporting_entry(
      analysis_project_file(), reporting_test_data_root(), output_root
    ),
    "frozen outputs.root"
  )
  expect_equal(readLines(manifest, warn = FALSE), "prior success")
})

test_that("reporting rejects a symbolic-link output alias", {
  alias <- tempfile("reporting-symlink-")
  expect_true(file.symlink(analysis_project_file("06_结果", "analysis"), alias))
  on.exit(if (file.exists(alias)) unlink(alias), add = TRUE)

  expect_error(
    run_reporting_entry(
      analysis_project_file(), reporting_test_data_root(), alias
    ),
    "symbolic link"
  )
})

test_that("reporting capacity gate runs before manifest revocation", {
  output_root <- tempfile("reporting-quota-")
  dir.create(output_root)
  manifest <- file.path(output_root, "run_manifest.json")
  writeLines("prior success", manifest)
  quota_gate <- function(...) stop("10 GB analysis output quota reached")

  expect_error(
    run_reporting_entry(
      analysis_project_file(), reporting_test_data_root(), output_root,
      preflight = quota_gate
    ),
    "10 GB analysis output quota"
  )
  expect_equal(readLines(manifest, warn = FALSE), "prior success")
})

test_that("staged publication validation rejects stale companions", {
  inputs <- load_reporting_inputs(analysis_project_file("06_结果", "analysis"))
  output_root <- tempfile("reporting-stage-")
  dir.create(output_root)
  render_reporting_outputs(inputs, output_root)
  writeLines("stale", file.path(output_root, "tables", "stale.csv"))
  expect_error(
    validate_staged_publication_set(output_root),
    "unexpected publication files"
  )
})
