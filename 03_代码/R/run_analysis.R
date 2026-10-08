script_argument <- grep("^--file=", commandArgs(FALSE), value = TRUE)
if (length(script_argument) != 1L) {
  stop("cannot resolve the analysis runner path")
}
script_path <- normalizePath(
  sub("^--file=", "", script_argument),
  mustWork = TRUE
)
project_root <- normalizePath(
  file.path(dirname(script_path), "..", ".."),
  mustWork = TRUE
)
Sys.setenv(
  GREEN_DEBT_PROJECT_ROOT = project_root,
  JULIACONNECTOR_JULIAOPTS = paste0(
    "--project=",
    file.path(project_root, "julia")
  )
)
.libPaths(c(file.path(project_root, ".r-bootstrap-library"), .libPaths()))
if (!requireNamespace("renv", quietly = TRUE)) {
  stop("renv is unavailable from the frozen bootstrap library")
}
renv::load(project = project_root)

source(file.path(project_root, "03_代码", "R", "00_utils.R"))
source(file.path(project_root, "03_代码", "R", "10_lp_models.R"))
source(file.path(project_root, "03_代码", "R", "11_inference.R"))
source(file.path(project_root, "03_代码", "R", "20_threshold.R"))
source(file.path(project_root, "03_代码", "R", "21_weak_iv.R"))
source(file.path(project_root, "03_代码", "R", "22_shift_share_audit.R"))
source(file.path(project_root, "03_代码", "R", "30_reporting.R"))

parse_analysis_arguments <- function(arguments) {
  kinds <- c("lp", "threshold", "weak-iv", "shift-share", "report")
  if (!length(arguments) || !arguments[[1L]] %in% kinds) {
    stop(
      "usage: run_analysis.R KIND --data-root PATH --output-root PATH; ",
      "KIND is lp, threshold, weak-iv, shift-share, or report"
    )
  }
  remaining <- arguments[-1L]
  expected_length <- 4L
  if (length(remaining) != expected_length) {
    stop("analysis requires exactly --data-root and --output-root")
  }
  result <- list(kind = arguments[[1L]])
  index <- 1L
  while (index <= length(remaining)) {
    option <- remaining[[index]]
    permitted <- c("--data-root", "--output-root")
    if (!option %in% permitted) {
      stop("unsupported analysis argument: ", option)
    }
    field <- sub("^--", "", option)
    field <- gsub("-", "_", field)
    if (!is.null(result[[field]])) {
      stop("duplicate analysis argument: ", option)
    }
    result[[field]] <- remaining[[index + 1L]]
    index <- index + 2L
  }
  result
}

read_json_object <- function(path, label) {
  if (!file.exists(path)) {
    stop(label, " is missing: ", path)
  }
  value <- jsonlite::read_json(path, simplifyVector = FALSE)
  if (!is.list(value) || is.null(names(value))) {
    stop(label, " must be a JSON object")
  }
  value
}

run_context_from_gate <- function(gate) {
  fields <- c(
    "run_id", "spec_id", "input_authority_hash", "git_commit",
    "renv_lock_sha256", "evidence_policy_sha256", "seed", "created_at_utc"
  )
  missing <- setdiff(fields, names(gate))
  if (length(missing)) {
    stop("analysis gate is missing: ", paste(missing, collapse = ", "))
  }
  context <- gate[fields]
  context$seed <- as.integer(context$seed)
  validate_run_context(context)
  context
}

require_context_match <- function(payload, context, label) {
  for (field in names(context)) {
    if (is.null(payload[[field]]) || !identical(
      as.character(payload[[field]]),
      as.character(context[[field]])
    )) {
      stop(label, " does not match run context: ", field)
    }
  }
  invisible(TRUE)
}

exact_model_sample_r <- function(panel, cell, period) {
  family <- as.character(cell$analysis_family)
  selected <- (
    panel$outcome_id == as.character(cell$outcome_id) &
      panel$horizon == as.integer(cell$horizon) &
      panel$gad_version == as.character(cell$gad_version) &
      panel$sample_version == as.character(cell$sample_version) &
      panel$treatment_time >= period[[1L]] &
      panel$treatment_time <= period[[2L]]
  )
  if (family %in% c("confirmatory", "bounded_controls")) {
    selected <- selected &
      panel$confirmatory_iv_eligible &
      !panel$descriptive_only &
      !panel$threshold_selection_only &
      !panel$negative_shock_sample
  } else if (family == "threshold_selection") {
    selected <- selected & panel$threshold_selection_only
  } else if (family == "vulnerability") {
    selected <- selected & panel$negative_shock_sample
  } else {
    stop("unsupported continuous analysis family: ", family)
  }
  selected[is.na(selected)] <- FALSE
  source_columns <- c(
    "economy_id",
    "treatment_time",
    "delta_outcome",
    "gimc_p01_p99",
    "gad_lag_p01_p99",
    "Z_p01_p99",
    "Z_GAD_p01_p99",
    "renewable_energy_consumption_share_analysis_p01_p99",
    "trade_openness_percent_gdp_analysis_p01_p99",
    "industry_value_added_share_analysis_p01_p99",
    "gdp_per_capita_current_usd_analysis_p01_p99"
  )
  d <- panel[selected, source_columns, drop = FALSE]
  names(d) <- c(
    "economy_id",
    "treatment_time",
    "delta_outcome",
    "gimc_a",
    "gad_a",
    "z_a",
    "z_gad_a",
    main_controls
  )
  d <- d[stats::complete.cases(d), , drop = FALSE]
  numeric_columns <- setdiff(names(d), "economy_id")
  if (any(!is.finite(as.matrix(d[numeric_columns])))) {
    stop("nonfinite frozen values in exact model sample")
  }
  d$gimc_gad_a <- d$gimc_a * d$gad_a
  d <- d[c(
    "economy_id", "treatment_time", "delta_outcome",
    "gimc_a", "gimc_gad_a", "gad_a", "z_a", "z_gad_a",
    main_controls
  )]
  d <- prepare_model_frame(d, cell)
  if (family == "vulnerability" && any(d$z_a >= 0)) {
    stop("negative-shock sample requires Z < 0 for every row")
  }
  d
}

require_sample_gate_match <- function(d, gate_cell) {
  observed <- c(
    n = nrow(d),
    economies = length(unique(as.character(d$economy_id))),
    clusters = length(unique(as.character(d$economy_id)))
  )
  expected <- c(
    n = as.integer(gate_cell$n),
    economies = as.integer(gate_cell$economies),
    clusters = as.integer(gate_cell$clusters)
  )
  if (!identical(as.integer(observed), as.integer(expected))) {
    stop(
      "exact R model sample does not match stage A gate: ",
      paste(names(observed), observed, expected, sep = "=", collapse = ", ")
    )
  }
  invisible(TRUE)
}

require_first_stage_match <- function(metrics, gate_cell, tolerance = 1e-8) {
  integer_fields <- c(
    "instrument_rank", "cross_moment_rank", "rank"
  )
  for (field in integer_fields) {
    if (as.integer(metrics[[field]]) != as.integer(gate_cell[[field]])) {
      stop("fail_cross_language_mismatch: ", field)
    }
  }
  numeric_fields <- c(
    "condition_number",
    "partial_r2_gimc",
    "partial_r2_interaction",
    "effective_f_gimc",
    "effective_f_interaction"
  )
  for (field in numeric_fields) {
    observed <- as.numeric(metrics[[field]])
    expected <- as.numeric(gate_cell[[field]])
    if (
      length(observed) != 1L || length(expected) != 1L ||
        !is.finite(observed) || !is.finite(expected) ||
        abs(observed - expected) > tolerance
    ) {
      stop(
        "fail_cross_language_mismatch: ",
        field,
        " observed=",
        format(observed, digits = 17),
        " expected=",
        format(expected, digits = 17)
      )
    }
  }
  status <- if (as.integer(metrics$rank) < 2L) {
    "fail_rank_deficient"
  } else if (min(
    as.numeric(metrics$effective_f_gimc),
    as.numeric(metrics$effective_f_interaction)
  ) < 10) {
    "weak_reference_below_10"
  } else {
    "adequate_reference_10"
  }
  if (!identical(status, as.character(gate_cell$first_stage_status))) {
    stop("fail_cross_language_mismatch: first_stage_status")
  }
  invisible(TRUE)
}

cell_key <- function(cell) {
  paste(
    cell$analysis_family,
    cell$outcome_id,
    as.integer(cell$horizon),
    cell$gad_version,
    cell$sample_version,
    sep = "\r"
  )
}

sort_gate_cells <- function(cells) {
  keys <- vapply(cells, cell_key, character(1L))
  cells[order(keys)]
}

model_row_base <- function(
  context, estimator, cell, d, clusters,
  reference_distribution = "cluster_t",
  reference_df = cluster_reference_df(clusters),
  ssc_config = FROZEN_SSC_CONFIG
) {
  c(list(
    run_id = context$run_id,
    spec_id = context$spec_id,
    input_authority_hash = context$input_authority_hash,
    git_commit = context$git_commit,
    renv_lock_sha256 = context$renv_lock_sha256,
    created_at_utc = context$created_at_utc
  ), frozen_result_metadata(
    context,
    d,
    clusters,
    reference_distribution = reference_distribution,
    reference_df = reference_df,
    ssc_config = ssc_config
  ), list(
    estimator = estimator,
    analysis_family = as.character(cell$analysis_family),
    outcome_id = as.character(cell$outcome_id),
    horizon = as.integer(cell$horizon),
    gad_version = as.character(cell$gad_version),
    sample_version = as.character(cell$sample_version)
  ))
}

result_identity_columns <- c(
  "run_id", "spec_id", "input_authority_hash", "git_commit",
  "renv_lock_sha256", "created_at_utc", "evidence_policy_sha256",
  "year_min", "year_max", "r_version", "python_version",
  "julia_version", "package_versions_json", "random_seed",
  "ssc_config", "reference_distribution", "reference_df"
)

blank_wild <- function() {
  list(
    estimate = NA_real_,
    std_error = NA_real_,
    conf_low = NA_real_,
    conf_high = NA_real_,
    p_value = NA_real_,
    draws = NA_integer_,
    interval_status = "not_required"
  )
}

coefficient_wild_results <- function(
  bundle, d, estimator, inference_status, seed, draws
) {
  results <- setNames(
    replicate(length(MODEL_TERMS), blank_wild(), simplify = FALSE),
    MODEL_TERMS
  )
  if (inference_status != "wild_bootstrap_required") {
    return(results)
  }
  for (term in MODEL_TERMS) {
    raw <- if (estimator == "lp_fe") {
      run_wild_fe(bundle$fit, term, seed = seed, draws = draws)
    } else {
      run_wild_iv(d, term, seed = seed, draws = draws)
    }
    results[[term]] <- extract_wild_result(raw)
  }
  results
}

estimate_rows_for_bundle <- function(
  bundle, gate_cell, inference_status, wild_results
) {
  rows <- vector("list", length(MODEL_TERMS))
  names(rows) <- MODEL_TERMS
  for (term in MODEL_TERMS) {
    estimate <- as.numeric(bundle$coefficients[[term]])
    std_error <- as.numeric(bundle$std_error[[term]])
    wild <- wild_results[[term]]
    row_inference_status <- if (
      inference_status == "wild_bootstrap_required" &&
        identical(wild$interval_status, "unbounded")
    ) {
      "wild_bootstrap_unbounded"
    } else {
      inference_status
    }
    row <- c(
      model_row_base(
        bundle$run_context,
        bundle$estimator,
        bundle$cell,
        bundle$data,
        bundle$clusters
      ),
      list(
        term = term,
        estimate = estimate,
        std_error = std_error,
        conf_low = as.numeric(bundle$conf_low[[term]]),
        conf_high = as.numeric(bundle$conf_high[[term]]),
        p_value = as.numeric(bundle$p_value[[term]]),
        wild_conf_low = wild$conf_low,
        wild_conf_high = wild$conf_high,
        wild_p_value = wild$p_value,
        wild_draws = wild$draws,
        wild_seed = if (is.na(wild$draws)) {
          NA_real_
        } else {
          as.numeric(bundle$run_context$seed)
        },
        n = as.integer(bundle$n),
        economies = as.integer(bundle$economies),
        clusters = as.integer(bundle$clusters),
        first_stage_status = as.character(
          gate_cell$first_stage_status
        ),
        inference_status = row_inference_status
      )
    )
    rows[[term]] <- as.data.frame(
      row,
      stringsAsFactors = FALSE,
      check.names = FALSE
    )
  }
  do.call(rbind, rows)
}

covariance_rows_for_bundle <- function(bundle) {
  rows <- list()
  index <- 1L
  for (term_i in MODEL_TERMS) {
    for (term_j in MODEL_TERMS) {
      row <- c(
        model_row_base(
          bundle$run_context,
          bundle$estimator,
          bundle$cell,
          bundle$data,
          bundle$clusters
        ),
        list(
          clusters = as.integer(bundle$clusters),
          term_i = term_i,
          term_j = term_j,
          covariance = as.numeric(
            bundle$covariance[term_i, term_j]
          )
        )
      )
      rows[[index]] <- as.data.frame(
        row,
        stringsAsFactors = FALSE,
        check.names = FALSE
      )
      index <- index + 1L
    }
  }
  do.call(rbind, rows)
}

marginal_rows_for_bundle <- function(
  bundle,
  d,
  gate_cell,
  inference_status,
  quantiles,
  seed,
  draws
) {
  gad_values <- as.numeric(stats::quantile(
    d$gad_a,
    probs = quantiles,
    names = FALSE,
    type = 7
  ))
  conventional <- compute_marginal_effects(
    bundle,
    gad_values = gad_values,
    gad_quantiles = quantiles
  )
  rows <- vector("list", length(quantiles))
  for (index in seq_along(quantiles)) {
    wild <- blank_wild()
    if (inference_status == "wild_bootstrap_required") {
      raw <- if (bundle$estimator == "lp_fe") {
        run_wild_fe_marginal(
          d,
          gad_values[[index]],
          seed = seed,
          draws = draws
        )
      } else {
        run_wild_iv_marginal(
          d,
          gad_values[[index]],
          seed = seed,
          draws = draws
        )
      }
      wild <- extract_wild_result(raw)
    }
    row_inference_status <- if (
      inference_status == "wild_bootstrap_required" &&
        identical(wild$interval_status, "unbounded")
    ) {
      "wild_bootstrap_unbounded"
    } else {
      inference_status
    }
    conventional_row <- conventional[index, , drop = FALSE]
    row <- c(
      model_row_base(
        bundle$run_context,
        bundle$estimator,
        bundle$cell,
        bundle$data,
        bundle$clusters
      ),
      list(
        gad_quantile = as.numeric(
          conventional_row$gad_quantile
        ),
        gad_value = as.numeric(conventional_row$gad_value),
        estimate = as.numeric(conventional_row$estimate),
        std_error = as.numeric(conventional_row$std_error),
        conf_low = as.numeric(conventional_row$conf_low),
        conf_high = as.numeric(conventional_row$conf_high),
        p_value = as.numeric(conventional_row$p_value),
        wild_conf_low = wild$conf_low,
        wild_conf_high = wild$conf_high,
        wild_p_value = wild$p_value,
        wild_draws = wild$draws,
        wild_seed = if (is.na(wild$draws)) {
          NA_real_
        } else {
          as.numeric(seed)
        },
        n = as.integer(bundle$n),
        economies = as.integer(bundle$economies),
        clusters = as.integer(bundle$clusters),
        first_stage_status = as.character(
          gate_cell$first_stage_status
        ),
        inference_status = row_inference_status
      )
    )
    rows[[index]] <- as.data.frame(
      row,
      stringsAsFactors = FALSE,
      check.names = FALSE
    )
  }
  do.call(rbind, rows)
}

bind_rows_exact <- function(rows, columns, label) {
  if (!length(rows)) {
    stop(label, " produced no rows")
  }
  result <- do.call(rbind, rows)
  missing <- setdiff(columns, names(result))
  extra <- setdiff(names(result), columns)
  if (length(missing) || length(extra)) {
    stop(label, " produced invalid columns")
  }
  result <- result[columns]
  rownames(result) <- NULL
  result
}

file_sha256 <- function(path) {
  digest::digest(file = path, algo = "sha256", serialize = FALSE)
}

audit_staging_root <- function(output_root, run_context) {
  file.path(
    output_root, "_staging", run_context$run_id,
    "threshold-and-iv-audit"
  )
}

write_audit_stage <- function(
  output_root, run_context, gate_path, stage, tables,
  registry = NULL
) {
  staging_root <- audit_staging_root(output_root, run_context)
  if (!dir.exists(staging_root) && !dir.create(staging_root, recursive = TRUE)) {
    stop("cannot create threshold-and-IV-audit staging directory")
  }
  receipt_path <- file.path(staging_root, "receipt.json")
  existing <- if (file.exists(receipt_path)) {
    read_json_object(receipt_path, "threshold-and-IV-audit receipt")
  } else {
    list()
  }
  if (length(existing)) {
    require_context_match(
      existing, run_context, "threshold-and-IV-audit receipt"
    )
    if (
      !identical(existing$kind, "threshold-and-iv-audit") ||
        !identical(existing$gate_sha256, file_sha256(gate_path))
    ) {
      stop("existing threshold-and-IV-audit receipt binding differs")
    }
  }
  files <- if (is.list(existing$files)) existing$files else list()
  for (name in names(tables)) {
    path <- file.path(staging_root, paste0(name, ".csv"))
    atomic_write_csv(tables[[name]], path)
    files[[name]] <- list(
      name = basename(path),
      rows = nrow(tables[[name]]),
      sha256 = file_sha256(path)
    )
  }
  completed <- sort(unique(c(
    unlist(existing$completed_stages), as.character(stage)
  )))
  binding_fields <- c(
    "registry_sha256", "registry_hash", "registry_sample_hash",
    "registry_q", "selection_outcome"
  )
  binding <- existing[intersect(binding_fields, names(existing))]
  if (!is.null(registry)) {
    registry_path <- file.path(
      output_root, "registries", "threshold_registry_v1.json"
    )
    binding <- list(
      registry_sha256 = file_sha256(registry_path),
      registry_hash = as.character(registry$registry_hash),
      registry_sample_hash = as.character(registry$sample_hash),
      registry_q = as.numeric(registry$q),
      selection_outcome = as.character(registry$selection_outcome)
    )
  }
  receipt <- c(
    run_context,
    list(
      kind = "threshold-and-iv-audit",
      gate_sha256 = file_sha256(gate_path),
      completed_stages = completed,
      files = files
    ),
    binding
  )
  atomic_write_json(receipt, receipt_path)
  staging_root
}

threshold_spec_from_yaml <- function(spec) {
  threshold <- spec$threshold
  endpoints <- as.integer(unlist(threshold$percentiles))
  threshold$percentiles <- seq(endpoints[[1L]], endpoints[[2L]])
  threshold$seed <- as.integer(spec$seed)
  threshold
}

threshold_failure_rows <- function(d, cell, context, registry, detail) {
  regime_index <- list(
    low = d$gad_a <= as.numeric(registry$q),
    high = d$gad_a > as.numeric(registry$q)
  )
  rows <- lapply(names(regime_index), function(regime) {
    row <- c(
      model_row_base(
        context, "threshold_iv", cell, d,
        length(unique(d$economy_id))
      ),
      list(
        selection_outcome = as.character(registry$selection_outcome),
        registry_hash = as.character(registry$registry_hash),
        registry_sample_hash = as.character(registry$sample_hash),
        q = as.numeric(registry$q),
        regime = regime,
        estimate = NA_real_, std_error = NA_real_,
        conf_low = NA_real_, conf_high = NA_real_, p_value = NA_real_,
        wild_estimate = NA_real_, wild_std_error = NA_real_,
        wild_conf_low = NA_real_, wild_conf_high = NA_real_,
        wild_p_value = NA_real_, wild_inference_status = "not_available",
        wild_draws = NA_integer_, wild_seed = NA_real_,
        low_high_covariance = NA_real_,
        difference_estimate = NA_real_, difference_std_error = NA_real_,
        difference_conf_low = NA_real_, difference_conf_high = NA_real_,
        difference_p_value = NA_real_,
        difference_wild_estimate = NA_real_,
        difference_wild_std_error = NA_real_,
        difference_wild_conf_low = NA_real_,
        difference_wild_conf_high = NA_real_,
        difference_wild_p_value = NA_real_,
        difference_wild_inference_status = "not_available",
        difference_wild_draws = NA_integer_,
        difference_wild_seed = NA_real_,
        regime_n = as.integer(sum(regime_index[[regime]])),
        regime_share = as.numeric(mean(regime_index[[regime]])),
        n = as.integer(nrow(d)),
        economies = as.integer(length(unique(d$economy_id))),
        clusters = as.integer(length(unique(d$economy_id))),
        first_stage_status = "fit_failed",
        inference_status = "fit_failed"
      )
    )
    as.data.frame(row, stringsAsFactors = FALSE, check.names = FALSE)
  })
  message(
    "threshold IV explicit failure: ", cell_key(cell), " ", detail
  )
  do.call(rbind, rows)
}

run_threshold_stage <- function(
  panel, gate, spec, run_context, output_root, gate_path
) {
  selection_cells <- Filter(
    function(cell) identical(
      as.character(cell$analysis_family), "threshold_selection"
    ),
    gate$cells
  )
  if (length(selection_cells) != 1L) {
    stop("threshold runner requires exactly one selection cell")
  }
  period <- as.integer(unlist(spec$period))
  selection_cell <- selection_cells[[1L]]
  selection_sample <- exact_model_sample_r(panel, selection_cell, period)
  require_sample_gate_match(selection_sample, selection_cell)
  registry_path <- file.path(
    output_root, "registries", "threshold_registry_v1.json"
  )
  registry <- select_and_freeze_threshold(
    selection_sample,
    registry_path,
    threshold_spec_from_yaml(spec),
    run_context,
    bootstrap_draws = 999L,
    minimum_valid_draws = 900L
  )
  cells <- sort_gate_cells(Filter(
    function(cell) identical(
      as.character(cell$analysis_family), "confirmatory"
    ),
    gate$cells
  ))
  if (length(cells) != 39L) {
    stop("threshold runner requires exactly 39 confirmatory cells")
  }
  rows <- vector("list", length(cells))
  for (index in seq_along(cells)) {
    cell <- cells[[index]]
    d <- exact_model_sample_r(panel, cell, period)
    require_sample_gate_match(d, cell)
    rows[[index]] <- tryCatch({
      bundle <- fit_threshold_iv(d, cell, run_context, registry$q)
      threshold_estimate_rows(
        bundle, d, registry,
        seed = as.integer(spec$seed),
        draws = as.integer(spec$inference$wild_bootstrap_draws)
      )
    }, error = function(error) {
      threshold_failure_rows(d, cell, run_context, registry, conditionMessage(error))
    })
    if (index %% 10L == 0L || index == length(cells)) {
      message("threshold IV progress: ", index, "/", length(cells))
    }
  }
  columns <- c(
    result_identity_columns, "estimator",
    "analysis_family", "selection_outcome", "registry_hash",
    "registry_sample_hash", "q", "outcome_id", "horizon",
    "gad_version", "sample_version", "regime", "estimate",
    "std_error", "conf_low", "conf_high", "p_value",
    "wild_estimate", "wild_std_error", "wild_conf_low", "wild_conf_high",
    "wild_p_value", "wild_inference_status", "wild_draws", "wild_seed",
    "low_high_covariance", "difference_estimate", "difference_std_error",
    "difference_conf_low", "difference_conf_high", "difference_p_value",
    "difference_wild_estimate", "difference_wild_std_error",
    "difference_wild_conf_low", "difference_wild_conf_high",
    "difference_wild_p_value", "difference_wild_inference_status",
    "difference_wild_draws", "difference_wild_seed", "regime_n",
    "regime_share", "n", "economies", "clusters",
    "first_stage_status", "inference_status"
  )
  estimates <- bind_rows_exact(rows, columns, "threshold estimates")
  staging <- write_audit_stage(
    output_root, run_context, gate_path, "threshold",
    list(threshold_estimates = estimates), registry
  )
  cat(jsonlite::toJSON(list(
    status = "staged", kind = "threshold", run_id = run_context$run_id,
    q = as.numeric(registry$q), percentile = as.integer(registry$percentile),
    rows = nrow(estimates), staging_path = staging
  ), auto_unbox = TRUE), "\n")
}

weak_iv_failure_row <- function(d, cell, context, detail) {
  message("weak-IV explicit failure: ", cell_key(cell), " ", detail)
  row <- c(
    model_row_base(
      context, "lp_iv_ar", cell, d, length(unique(d$economy_id)),
      reference_distribution = "cr2_htz_f",
      reference_df = cluster_reference_df(length(unique(d$economy_id))),
      ssc_config = "clubSandwich:CR2;Wald_test=HTZ"
    ),
    list(
      beta_low = NA_real_, beta_high = NA_real_,
      theta_low = NA_real_, theta_high = NA_real_,
      status = "unavailable", expansions = 0L,
      accepted_points = 0L,
      accepted_hash = canonical_frame_sha256(data.frame(
        beta = numeric(), theta = numeric()
      )),
      conventional_point_accepted = FALSE,
      n = as.integer(nrow(d)),
      economies = as.integer(length(unique(d$economy_id))),
      clusters = as.integer(length(unique(d$economy_id))),
      inference_status = "unavailable"
    )
  )
  as.data.frame(row, stringsAsFactors = FALSE, check.names = FALSE)
}

run_weak_iv_stage <- function(
  panel, gate, spec, run_context, output_root, gate_path
) {
  cells <- sort_gate_cells(Filter(function(cell) {
    as.character(cell$analysis_family) %in% c("confirmatory", "vulnerability")
  }, gate$cells))
  if (length(cells) != 45L) {
    stop("weak-IV runner requires exactly 45 confirmatory/vulnerability cells")
  }
  period <- as.integer(unlist(spec$period))
  ar_spec <- spec$inference
  ar_spec$alpha <- 0.05
  rows <- vector("list", length(cells))
  for (index in seq_along(cells)) {
    cell <- cells[[index]]
    d <- exact_model_sample_r(panel, cell, period)
    require_sample_gate_match(d, cell)
    rows[[index]] <- tryCatch({
      conventional <- fit_lp_iv(d, cell, run_context)
      result <- compute_ar_region(d, conventional, cell, ar_spec)
      row <- c(
        model_row_base(
          run_context, "lp_iv_ar", cell, d, result$clusters,
          reference_distribution = "cr2_htz_f",
          reference_df = result$reference_df,
          ssc_config = "clubSandwich:CR2;Wald_test=HTZ"
        ),
        list(
          beta_low = result$beta_conf_low,
          beta_high = result$beta_conf_high,
          theta_low = result$theta_conf_low,
          theta_high = result$theta_conf_high,
          status = result$status,
          expansions = result$expansions,
          accepted_points = result$accepted_points,
          accepted_hash = result$accepted_hash,
          conventional_point_accepted = result$conventional_point_accepted,
          n = result$n,
          economies = result$clusters,
          clusters = result$clusters,
          inference_status = "available"
        )
      )
      as.data.frame(row, stringsAsFactors = FALSE, check.names = FALSE)
    }, error = function(error) {
      weak_iv_failure_row(d, cell, run_context, conditionMessage(error))
    })
    if (index %% 5L == 0L || index == length(cells)) {
      message("weak-IV progress: ", index, "/", length(cells))
    }
  }
  columns <- c(
    result_identity_columns, "estimator",
    "analysis_family", "outcome_id", "horizon", "gad_version",
    "sample_version", "beta_low", "beta_high", "theta_low",
    "theta_high", "status", "expansions", "accepted_points",
    "accepted_hash", "conventional_point_accepted", "n", "economies",
    "clusters", "inference_status"
  )
  weak_sets <- bind_rows_exact(rows, columns, "weak-IV sets")
  staging <- write_audit_stage(
    output_root, run_context, gate_path, "weak-iv",
    list(weak_iv_sets = weak_sets)
  )
  cat(jsonlite::toJSON(list(
    status = "staged", kind = "weak-iv", run_id = run_context$run_id,
    rows = nrow(weak_sets), staging_path = staging
  ), auto_unbox = TRUE), "\n")
}

load_main_partner_shocks <- function(path, importers, period) {
  dataset <- arrow::open_dataset(path, format = "parquet")
  query <- dplyr::filter(
    dataset,
    taxonomy_version == "main_hs96",
    share_version == "main_0.0001",
    destination_excluded == importer,
    importer %in% importers,
    year >= period[[1L]], year <= period[[2L]],
    !is.na(contribution)
  )
  query <- dplyr::select(
    query, importer, exporter, hs6, year, contribution
  )
  result <- as.data.frame(dplyr::collect(query))
  if (!nrow(result)) {
    stop("filtered main partner shocks are empty")
  }
  result$importer <- as.character(result$importer)
  result$exporter <- as.character(result$exporter)
  result$hs6 <- as.character(result$hs6)
  result$year <- as.integer(result$year)
  result
}

shift_share_summary_row <- function(audit, d, cell, context) {
  summary <- audit$summary
  row <- c(
    model_row_base(
      context, "lp_iv_shift_share", cell, d,
      max(2L, as.integer(summary$shock_clusters)),
      reference_distribution = "not_used",
      reference_df = max(1, as.numeric(summary$shock_clusters) - 1),
      ssc_config = "shock_cluster_sandwich:exporter_hs6;reference=not_used"
    ),
    list(
      n = as.integer(nrow(d)),
      economies = as.integer(length(unique(d$economy_id))),
      shock_observations = summary$shock_observations,
      shock_clusters = summary$shock_clusters,
      signed_weight_sum = summary$signed_weight_sum,
      absolute_weight_sum = summary$absolute_weight_sum,
      hhi_absolute = summary$hhi_absolute,
      top1_absolute_share = summary$top1_absolute_share,
      top5_absolute_share = summary$top5_absolute_share,
      negative_weight_share = summary$negative_weight_share,
      z_reconstruction_error = summary$z_reconstruction_error,
      z_gad_reconstruction_error = summary$z_gad_reconstruction_error,
      shock_std_error_gimc = summary$shock_std_error_gimc,
      shock_std_error_interaction = summary$shock_std_error_interaction,
      cross_moment_rank = summary$cross_moment_rank,
      shock_inference_status = summary$shock_inference_status
    )
  )
  as.data.frame(row, stringsAsFactors = FALSE, check.names = FALSE)
}

shift_share_weight_rows <- function(audit, d, cell, context) {
  if (!nrow(audit$weights)) {
    return(NULL)
  }
  selected <- head(audit$weights, 100L)
  rows <- lapply(seq_len(nrow(selected)), function(index) {
    weight <- selected[index, , drop = FALSE]
    base <- model_row_base(
      context, "lp_iv_shift_share", cell, d,
      max(2L, as.integer(audit$summary$shock_clusters)),
      reference_distribution = "not_used",
      reference_df = max(1, as.numeric(audit$summary$shock_clusters) - 1),
      ssc_config = "shock_cluster_sandwich:exporter_hs6;reference=not_used"
    )
    base$estimator <- NULL
    row <- c(
      base,
      list(
        shock_id = as.character(weight$shock_id),
        exporter = as.character(weight$exporter),
        hs6 = as.character(weight$hs6),
        year = as.integer(weight$year),
        shock_cluster_id = as.character(weight$shock_cluster_id),
        signed_weight = as.numeric(weight$signed_weight),
        absolute_weight = as.numeric(weight$absolute_weight),
        absolute_rank = as.integer(weight$absolute_rank)
      )
    )
    as.data.frame(row, stringsAsFactors = FALSE, check.names = FALSE)
  })
  do.call(rbind, rows)
}

run_shift_share_stage <- function(
  panel, gate, spec, run_context, data_root, output_root, gate_path
) {
  cells <- sort_gate_cells(Filter(function(cell) {
    as.character(cell$analysis_family) %in% c("confirmatory", "vulnerability")
  }, gate$cells))
  if (length(cells) != 45L) {
    stop("shift-share runner requires exactly 45 confirmatory/vulnerability cells")
  }
  period <- as.integer(unlist(spec$period))
  importers <- sort(unique(as.character(panel$economy_id)))
  shock_path <- file.path(
    data_root, "05_中间数据", "measures", "instruments",
    "iv_partner_shocks.parquet"
  )
  shocks <- load_main_partner_shocks(shock_path, importers, period)
  shocks$model_key <- paste(shocks$importer, shocks$year, sep = "\r")
  summary_rows <- vector("list", length(cells))
  weight_rows <- list()
  weight_index <- 1L
  for (index in seq_along(cells)) {
    cell <- cells[[index]]
    d <- exact_model_sample_r(panel, cell, period)
    require_sample_gate_match(d, cell)
    conventional <- fit_lp_iv(d, cell, run_context)
    residualized <- qr_residualize(
      as.matrix(d[c("delta_outcome", "gimc_a", "gimc_gad_a")]),
      fixed_effect_design(d)
    )
    structural_residual <- as.numeric(
      residualized[, 1L] - residualized[, 2:3, drop = FALSE] %*%
        as.numeric(conventional$coefficients[c("gimc_a", "gimc_gad_a")])
    )
    model_keys <- data.frame(
      economy_id = d$economy_id,
      treatment_time = d$treatment_time,
      outcome_id = as.character(cell$outcome_id),
      horizon = as.integer(cell$horizon),
      gad_version = as.character(cell$gad_version),
      sample_version = as.character(cell$sample_version),
      gimc_a = residualized[, 2L],
      gimc_gad_a = residualized[, 3L],
      residual = structural_residual,
      stringsAsFactors = FALSE
    )
    audit_sample <- exact_shift_share_sample(panel, model_keys)
    sample_keys <- unique(paste(
      audit_sample$economy_id, audit_sample$treatment_time, sep = "\r"
    ))
    cell_shocks <- shocks[shocks$model_key %in% sample_keys, , drop = FALSE]
    audit <- generalized_rotemberg(list(
      sample = audit_sample,
      shocks = cell_shocks
    ))
    summary_rows[[index]] <- shift_share_summary_row(
      audit, d, cell, run_context
    )
    cell_weights <- shift_share_weight_rows(audit, d, cell, run_context)
    if (!is.null(cell_weights)) {
      weight_rows[[weight_index]] <- cell_weights
      weight_index <- weight_index + 1L
    }
    if (index %% 5L == 0L || index == length(cells)) {
      message("shift-share progress: ", index, "/", length(cells))
    }
  }
  summary_columns <- c(
    result_identity_columns, "estimator",
    "analysis_family", "outcome_id", "horizon", "gad_version",
    "sample_version", "n", "economies", "shock_observations",
    "shock_clusters", "signed_weight_sum", "absolute_weight_sum",
    "hhi_absolute", "top1_absolute_share", "top5_absolute_share",
    "negative_weight_share", "z_reconstruction_error",
    "z_gad_reconstruction_error", "shock_std_error_gimc",
    "shock_std_error_interaction", "cross_moment_rank",
    "shock_inference_status"
  )
  weight_columns <- c(
    result_identity_columns, "analysis_family",
    "outcome_id", "horizon", "gad_version", "sample_version",
    "shock_id", "exporter", "hs6", "year", "shock_cluster_id",
    "signed_weight", "absolute_weight", "absolute_rank"
  )
  summaries <- bind_rows_exact(
    summary_rows, summary_columns, "shift-share summaries"
  )
  weights <- bind_rows_exact(
    weight_rows, weight_columns, "shift-share top weights"
  )
  staging <- write_audit_stage(
    output_root, run_context, gate_path, "shift-share",
    list(
      shift_share_summary = summaries,
      shift_share_weights = weights
    )
  )
  cat(jsonlite::toJSON(list(
    status = "staged", kind = "shift-share", run_id = run_context$run_id,
    summary_rows = nrow(summaries), weight_rows = nrow(weights),
    staging_path = staging
  ), auto_unbox = TRUE), "\n")
}

arguments <- parse_analysis_arguments(commandArgs(trailingOnly = TRUE))
if (arguments$kind == "report") {
  result <- run_reporting_entry(
    project_root, arguments$data_root, arguments$output_root
  )
  cat(jsonlite::toJSON(list(
    status = "rendered", kind = "report",
    tables = length(result$tables), figures = length(result$figures)
  ), auto_unbox = TRUE), "\n")
  quit(save = "no", status = 0L)
}
reporting_analysis_preflight(
  project_root, arguments$data_root, arguments$output_root
)
output_root <- normalizePath(arguments$output_root, mustWork = TRUE)
data_root <- normalizePath(arguments$data_root, mustWork = TRUE)

spec <- yaml::read_yaml(file.path(project_root, "config", "analysis.yaml"))
gate_path <- file.path(
  output_root, "registries", "analysis_gate_v1.json"
)
summary_path <- file.path(
  output_root, "diagnostics", "stage_a_summary.json"
)
gate <- read_json_object(gate_path, "analysis gate")
summary <- read_json_object(summary_path, "stage A summary")
if (!identical(as.character(gate$status), "frozen")) {
  stop("analysis gate is not frozen")
}
if (!identical(as.character(summary$status), "valid")) {
  stop("stage A summary is not valid")
}
run_context <- run_context_from_gate(gate)
require_context_match(summary, run_context, "stage A summary")
if (
  !identical(as.character(spec$spec_id), run_context$spec_id) ||
    as.integer(spec$seed) != run_context$seed
) {
  stop("frozen R analysis spec does not match stage A")
}

panel_path <- file.path(
  data_root, "05_中间数据", "analysis", "lp_panel.parquet"
)
panel <- as.data.frame(arrow::read_parquet(panel_path))
required_panel_columns <- c(
  "economy_id",
  "treatment_time",
  "horizon",
  "outcome_id",
  "gad_version",
  "sample_version",
  "delta_outcome",
  "gimc_p01_p99",
  "gad_lag",
  "gad_lag_p01_p99",
  "Z",
  "Z_p01_p99",
  "Z_GAD",
  "Z_GAD_p01_p99",
  "confirmatory_iv_eligible",
  "descriptive_only",
  "threshold_selection_only",
  "negative_shock_sample",
  "renewable_energy_consumption_share_analysis_p01_p99",
  "trade_openness_percent_gdp_analysis_p01_p99",
  "industry_value_added_share_analysis_p01_p99",
  "gdp_per_capita_current_usd_analysis_p01_p99"
)
missing_panel_columns <- setdiff(required_panel_columns, names(panel))
if (length(missing_panel_columns)) {
  stop(
    "frozen model panel is missing: ",
    paste(missing_panel_columns, collapse = ", ")
  )
}

if (arguments$kind == "threshold") {
  run_threshold_stage(
    panel, gate, spec, run_context, output_root, gate_path
  )
  quit(save = "no", status = 0L)
}
if (arguments$kind == "weak-iv") {
  run_weak_iv_stage(
    panel, gate, spec, run_context, output_root, gate_path
  )
  quit(save = "no", status = 0L)
}
if (arguments$kind == "shift-share") {
  run_shift_share_stage(
    panel, gate, spec, run_context, data_root, output_root, gate_path
  )
  quit(save = "no", status = 0L)
}
if (arguments$kind != "lp") {
  stop("analysis runner reached an unsupported dispatch state")
}

continuous_cells <- Filter(
  function(cell) {
    !identical(
      as.character(cell$analysis_family),
      "threshold_selection"
    )
  },
  gate$cells
)
continuous_cells <- sort_gate_cells(continuous_cells)
if (length(continuous_cells) != 84L) {
  stop("LP runner requires exactly 84 continuous cells")
}
if (any(vapply(
  continuous_cells,
  function(cell) {
    clusters <- as.integer(cell$clusters)
    rank <- as.integer(cell$rank)
    rank >= 2L && clusters >= 20L && clusters < 30L
  },
  logical(1L)
))) {
  validate_julia_runtime(project_root)
}

estimate_columns <- c(
  result_identity_columns, "estimator",
  "analysis_family", "outcome_id", "horizon", "gad_version",
  "sample_version", "term", "estimate", "std_error",
  "conf_low", "conf_high", "p_value", "wild_conf_low",
  "wild_conf_high", "wild_p_value", "wild_draws", "wild_seed",
  "n", "economies", "clusters", "first_stage_status",
  "inference_status"
)
covariance_columns <- c(
  result_identity_columns, "estimator",
  "analysis_family", "outcome_id", "horizon", "gad_version",
  "sample_version", "clusters", "term_i", "term_j", "covariance"
)
marginal_columns <- c(
  result_identity_columns, "estimator",
  "analysis_family", "outcome_id", "horizon", "gad_version",
  "sample_version", "gad_quantile", "gad_value", "estimate",
  "std_error", "conf_low", "conf_high", "p_value", "wild_conf_low",
  "wild_conf_high", "wild_p_value", "wild_draws", "wild_seed",
  "n", "economies", "clusters", "first_stage_status",
  "inference_status"
)

fe_rows <- list()
iv_rows <- list()
covariance_rows <- list()
marginal_rows <- list()
skipped_cells <- list()
fe_index <- iv_index <- covariance_index <- marginal_index <- 1L
period <- as.integer(unlist(spec$period))
quantiles <- as.numeric(unlist(
  spec$inference$marginal_gad_quantiles
))
draws <- as.integer(spec$inference$wild_bootstrap_draws)

for (cell_index in seq_along(continuous_cells)) {
  gate_cell <- continuous_cells[[cell_index]]
  d <- exact_model_sample_r(panel, gate_cell, period)
  require_sample_gate_match(d, gate_cell)
  metrics <- compute_first_stage_metrics(d)
  require_first_stage_match(metrics, gate_cell)
  inference_status <- classify_inference(
    as.integer(gate_cell$clusters),
    as.integer(gate_cell$rank)
  )

  fe_bundle <- fit_lp_fe(d, gate_cell, run_context)
  fe_wild <- coefficient_wild_results(
    fe_bundle,
    d,
    "lp_fe",
    inference_status,
    run_context$seed,
    draws
  )
  fe_rows[[fe_index]] <- estimate_rows_for_bundle(
    fe_bundle,
    gate_cell,
    inference_status,
    fe_wild
  )
  fe_index <- fe_index + 1L
  covariance_rows[[covariance_index]] <-
    covariance_rows_for_bundle(fe_bundle)
  covariance_index <- covariance_index + 1L
  marginal_rows[[marginal_index]] <- marginal_rows_for_bundle(
    fe_bundle,
    d,
    gate_cell,
    inference_status,
    quantiles,
    run_context$seed,
    draws
  )
  marginal_index <- marginal_index + 1L

  if (as.integer(gate_cell$rank) < 2L) {
    skipped_cells[[length(skipped_cells) + 1L]] <- list(
      estimator = "lp_iv",
      analysis_family = gate_cell$analysis_family,
      outcome_id = gate_cell$outcome_id,
      horizon = as.integer(gate_cell$horizon),
      gad_version = gate_cell$gad_version,
      sample_version = gate_cell$sample_version,
      status = "fail_rank_deficient"
    )
  } else {
    iv_bundle <- fit_lp_iv(d, gate_cell, run_context)
    iv_wild <- coefficient_wild_results(
      iv_bundle,
      d,
      "lp_iv",
      inference_status,
      run_context$seed,
      draws
    )
    iv_rows[[iv_index]] <- estimate_rows_for_bundle(
      iv_bundle,
      gate_cell,
      inference_status,
      iv_wild
    )
    iv_index <- iv_index + 1L
    covariance_rows[[covariance_index]] <-
      covariance_rows_for_bundle(iv_bundle)
    covariance_index <- covariance_index + 1L
    marginal_rows[[marginal_index]] <- marginal_rows_for_bundle(
      iv_bundle,
      d,
      gate_cell,
      inference_status,
      quantiles,
      run_context$seed,
      draws
    )
    marginal_index <- marginal_index + 1L
  }

  if (cell_index %% 10L == 0L || cell_index == length(continuous_cells)) {
    message(
      "LP progress: ",
      cell_index,
      "/",
      length(continuous_cells)
    )
  }
}

lp_fe <- bind_rows_exact(fe_rows, estimate_columns, "LP-FE")
lp_iv <- bind_rows_exact(iv_rows, estimate_columns, "LP-IV")
model_covariance <- bind_rows_exact(
  covariance_rows,
  covariance_columns,
  "model covariance"
)
marginal_effects <- bind_rows_exact(
  marginal_rows,
  marginal_columns,
  "marginal effects"
)

staging_root <- file.path(
  output_root, "_staging", run_context$run_id, "lp"
)
if (!dir.exists(staging_root) && !dir.create(
  staging_root,
  recursive = TRUE
)) {
  stop("cannot create LP staging directory")
}
tables <- list(
  lp_fe = lp_fe,
  lp_iv = lp_iv,
  model_covariance = model_covariance,
  marginal_effects = marginal_effects
)
files <- list()
for (name in names(tables)) {
  path <- file.path(staging_root, paste0(name, ".csv"))
  atomic_write_csv(tables[[name]], path)
  files[[name]] <- list(
    name = basename(path),
    rows = nrow(tables[[name]]),
    sha256 = file_sha256(path)
  )
}
receipt <- c(
  run_context,
  list(
    kind = "lp",
    gate_sha256 = file_sha256(gate_path),
    files = files,
    skipped_cells = skipped_cells
  )
)
receipt_path <- file.path(staging_root, "receipt.json")
atomic_write_json(receipt, receipt_path)

cat(jsonlite::toJSON(
  list(
    status = "staged",
    kind = "lp",
    run_id = run_context$run_id,
    continuous_cells = length(continuous_cells),
    lp_fe_rows = nrow(lp_fe),
    lp_iv_rows = nrow(lp_iv),
    covariance_rows = nrow(model_covariance),
    marginal_rows = nrow(marginal_effects),
    staging_path = staging_root
  ),
  auto_unbox = TRUE
), "\n")
