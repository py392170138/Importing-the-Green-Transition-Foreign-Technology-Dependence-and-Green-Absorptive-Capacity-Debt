reporting_table_paths <- c(
  sample_cells = "diagnostics/sample_cells.parquet",
  missingness = "diagnostics/missingness.parquet",
  distributions = "diagnostics/distributions.parquet",
  correlations = "diagnostics/correlations.parquet",
  descriptive_paths = "diagnostics/descriptive_paths.parquet",
  exposure_concentration = "diagnostics/exposure_concentration.parquet",
  first_stage_screen = "diagnostics/first_stage_screen.parquet",
  shift_share_summary = "diagnostics/shift_share_summary.parquet",
  shift_share_weights = "diagnostics/shift_share_weights.parquet",
  lp_fe = "models/lp_fe.parquet",
  lp_iv = "models/lp_iv.parquet",
  model_covariance = "models/model_covariance.parquet",
  marginal_effects = "models/marginal_effects.parquet",
  threshold_estimates = "models/threshold_estimates.parquet",
  weak_iv_sets = "models/weak_iv_sets.parquet"
)

reporting_git_commit <- function() {
  root <- Sys.getenv("GREEN_DEBT_PROJECT_ROOT", unset = NA_character_)
  if (is.na(root) || !nzchar(root)) stop("GREEN_DEBT_PROJECT_ROOT is required")
  value <- system2(
    "git", c("-C", shQuote(root), "rev-parse", "HEAD"),
    stdout = TRUE, stderr = TRUE
  )
  status <- attr(value, "status")
  if (!is.null(status) && status != 0L) stop("cannot bind reporting Git commit")
  value <- trimws(paste(value, collapse = ""))
  if (!grepl("^[0-9a-f]{40}$", value)) stop("invalid reporting Git commit")
  value
}

reporting_analysis_preflight <- function(
  project_root, data_root, output_root
) {
  python <- file.path(project_root, ".venv", "bin", "python")
  if (!file.exists(python)) stop("frozen analysis Python runtime is missing")
  arguments <- c(
    "-m", "green_debt.cli", "analysis-preflight",
    "--data-root", data_root, "--output-root", output_root
  )
  output <- suppressWarnings(system2(
    python, shQuote(arguments), stdout = TRUE, stderr = TRUE
  ))
  status <- attr(output, "status")
  if (!is.null(status) && status != 0L) {
    stop(paste(output, collapse = "\n"), call. = FALSE)
  }
  if (!length(output)) stop("analysis preflight returned no receipt")
  receipt <- tryCatch(
    jsonlite::fromJSON(output[[length(output)]], simplifyVector = TRUE),
    error = function(error) stop("invalid analysis preflight receipt", call. = FALSE)
  )
  if (!identical(as.character(receipt$status), "ready")) {
    stop("analysis preflight did not report ready")
  }
  invisible(receipt)
}

run_reporting_entry <- function(
  project_root, data_root, output_root,
  preflight = reporting_analysis_preflight
) {
  preflight(project_root, data_root, output_root)
  normalized_output <- normalizePath(output_root, mustWork = TRUE)
  run_reporting(normalized_output)
}

build_main_tables <- function(inputs) {
  required <- c("outcome_id", "horizon", "term", "estimate", "std_error")
  missing <- setdiff(required, names(inputs$estimates))
  if (length(missing)) {
    stop("reporting estimates are missing: ", paste(missing, collapse = ", "))
  }
  selected <- inputs$estimates[
    inputs$estimates$term == "gimc_a",
    c("outcome_id", "horizon", "estimate", "std_error"),
    drop = FALSE
  ]
  selected <- selected[order(selected$outcome_id, selected$horizon), , drop = FALSE]
  rownames(selected) <- NULL
  list(lp_iv = selected)
}

canonical_reporting_data_hash <- function(data) {
  normalized <- data
  numeric_columns <- vapply(normalized, is.numeric, logical(1L))
  normalized[numeric_columns] <- lapply(
    normalized[numeric_columns],
    function(value) {
      finite <- is.finite(value)
      value[finite & abs(value) < 5e-13] <- 0
      value[finite] <- signif(value[finite], digits = 13L)
      value
    }
  )
  if (nrow(normalized) > 1L && ncol(normalized) > 0L) {
    row_order <- do.call(
      order,
      c(unname(normalized), list(na.last = TRUE, method = "radix"))
    )
    normalized <- normalized[row_order, , drop = FALSE]
  }
  rownames(normalized) <- NULL
  digest::digest(normalized, algo = "sha256")
}

build_dynamic_path_figure <- function(inputs) {
  data <- build_main_tables(inputs)$lp_iv
  source_hash <- canonical_reporting_data_hash(data)
  plot <- ggplot2::ggplot(
    data,
    ggplot2::aes(x = horizon, y = estimate, group = outcome_id)
  ) +
    ggplot2::geom_line() +
    ggplot2::geom_point()
  attr(plot, "source_data_hash") <- source_hash
  list(data = data, plot = plot)
}

load_reporting_inputs <- function(output_root) {
  root <- normalizePath(output_root, mustWork = TRUE)
  inputs <- list()
  hashes <- list()
  for (name in names(reporting_table_paths)) {
    path <- file.path(root, reporting_table_paths[[name]])
    manifest_path <- paste0(path, ".manifest.json")
    if (!file.exists(path) || !file.exists(manifest_path)) {
      stop("validated reporting input is missing: ", path)
    }
    manifest <- jsonlite::read_json(manifest_path, simplifyVector = TRUE)
    observed_hash <- digest::digest(file = path, algo = "sha256", serialize = FALSE)
    if (!identical(observed_hash, as.character(manifest$output_sha256))) {
      stop("reporting input hash mismatch: ", path)
    }
    inputs[[name]] <- as.data.frame(arrow::read_parquet(path))
    hashes[[name]] <- observed_hash
  }
  registry_path <- file.path(root, "registries", "threshold_registry_v1.json")
  inputs$threshold_registry <- jsonlite::read_json(
    registry_path, simplifyVector = FALSE
  )
  hashes$threshold_registry <- digest::digest(
    file = registry_path, algo = "sha256", serialize = FALSE
  )
  identity_fields <- c(
    "run_id", "spec_id", "input_authority_hash", "git_commit",
    "renv_lock_sha256", "created_at_utc"
  )
  identities <- lapply(inputs[names(reporting_table_paths)], function(value) {
    unique(value[c(
      identity_fields
    )])
  })
  reference <- identities[[1L]]
  if (any(!vapply(identities, identical, logical(1L), reference))) {
    stop("reporting inputs do not share one model identity")
  }
  inputs$source_table_hashes <- hashes
  inputs$run_context <- as.list(reference[1L, , drop = FALSE])
  result_tables <- c(
    "shift_share_summary", "shift_share_weights", "lp_fe", "lp_iv",
    "model_covariance", "marginal_effects", "threshold_estimates",
    "weak_iv_sets"
  )
  policy_hashes <- vapply(result_tables, function(name) {
    value <- inputs[[name]]
    if (!"evidence_policy_sha256" %in% names(value)) {
      stop("machine-readable result is missing evidence policy identity: ", name)
    }
    observed <- unique(as.character(value$evidence_policy_sha256))
    if (length(observed) != 1L || is.na(observed) || !nzchar(observed)) {
      stop("machine-readable result has ambiguous evidence policy identity: ", name)
    }
    observed
  }, character(1L))
  if (length(unique(policy_hashes)) != 1L) {
    stop("machine-readable results do not share one evidence policy identity")
  }
  inputs$run_context$evidence_policy_sha256 <- unname(policy_hashes[[1L]])
  project_root <- Sys.getenv("GREEN_DEBT_PROJECT_ROOT", unset = NA_character_)
  if (is.na(project_root) || !nzchar(project_root)) {
    stop("GREEN_DEBT_PROJECT_ROOT is required for evidence policy authority")
  }
  policy_path <- file.path(project_root, "config", "evidence_policy.json")
  inputs$evidence_policy <- jsonlite::read_json(
    policy_path, simplifyVector = FALSE
  )
  inputs$evidence_policy_sha256 <- digest::digest(
    file = policy_path, algo = "sha256", serialize = FALSE
  )
  if (!identical(
    inputs$evidence_policy_sha256,
    as.character(inputs$run_context$evidence_policy_sha256)
  )) {
    stop("reporting evidence policy hash differs from model identity")
  }
  inputs
}

bind_rows_fill <- function(...) {
  values <- list(...)
  columns <- unique(unlist(lapply(values, names), use.names = FALSE))
  normalized <- lapply(values, function(value) {
    missing <- setdiff(columns, names(value))
    for (column in missing) value[[column]] <- NA
    value[columns]
  })
  do.call(rbind, normalized)
}

metric_rows <- function(data, identifiers, metrics, section) {
  rows <- lapply(metrics, function(metric) {
    value <- data[identifiers]
    value$metric <- metric
    value$value <- data[[metric]]
    value
  })
  result <- do.call(rbind, rows)
  result$section <- section
  result[c("section", identifiers, "metric", "value")]
}

select_publication_intervals <- function(data) {
  required <- c(
    "clusters", "inference_status", "conf_low", "conf_high",
    "wild_conf_low", "wild_conf_high"
  )
  missing <- setdiff(required, names(data))
  if (length(missing)) {
    stop("publication interval inputs are missing: ", paste(missing, collapse = ", "))
  }
  result <- data
  result$publication_conf_low <- NA_real_
  result$publication_conf_high <- NA_real_
  result$publication_interval_status <- NA_character_
  exploratory <- result$clusters < 20L
  wild <- result$clusters >= 20L & result$clusters < 30L
  robust <- result$clusters >= 30L
  if (any(exploratory & result$inference_status != "exploratory_lt20_clusters")) {
    stop("clusters below 20 must be marked exploratory")
  }
  if (any(wild & !result$inference_status %in% c(
    "wild_bootstrap_required", "wild_bootstrap_unbounded"
  ))) {
    stop("clusters from 20 through 29 must use registered wild-bootstrap inference")
  }
  if (any(robust & result$inference_status != "cluster_robust")) {
    stop("30 or more clusters must use cluster-robust inference")
  }
  bounded_wild <- wild & result$inference_status == "wild_bootstrap_required"
  if (any(bounded_wild & (!is.finite(result$wild_conf_low) |
      !is.finite(result$wild_conf_high)))) {
    stop("registered wild-bootstrap intervals are missing")
  }
  result$publication_interval_status[exploratory] <- "point_only_exploratory"
  result$publication_conf_low[bounded_wild] <- result$wild_conf_low[bounded_wild]
  result$publication_conf_high[bounded_wild] <- result$wild_conf_high[bounded_wild]
  result$publication_interval_status[bounded_wild] <- "wild_bootstrap"
  unbounded_wild <- wild & result$inference_status == "wild_bootstrap_unbounded"
  result$publication_interval_status[unbounded_wild] <- "wild_bootstrap_unbounded"
  result$publication_conf_low[robust] <- result$conf_low[robust]
  result$publication_conf_high[robust] <- result$conf_high[robust]
  result$publication_interval_status[robust] <- "cluster_robust"
  result
}

publication_warning_labels <- function(data) {
  labels <- rep("CONVENTIONAL CLUSTER-t (30+ CLUSTERS)", nrow(data))
  labels[data$clusters < 20L] <- "POINT-ONLY/EXPLORATORY (<20 CLUSTERS)"
  low_cluster <- data$clusters >= 20L & data$clusters < 30L
  labels[low_cluster] <- "LOW-CLUSTER WILD BOOTSTRAP (20-29 CLUSTERS)"
  unbounded <- grepl("unbounded", data$inference_status, fixed = TRUE)
  labels[unbounded] <- "POINT-ONLY/UNBOUNDED WILD INTERVAL"
  labels
}

reporting_evidence_projection <- function(evidence_policy) {
  status <- as.character(evidence_policy$concentration$status)
  if (length(status) != 1L || !identical(
    status, "unresolved_no_preregistered_cutoff"
  )) {
    stop("reporting requires the frozen unresolved concentration authority")
  }
  grade <- as.character(evidence_policy$concentration$unresolved_policy)
  if (length(grade) != 1L || !identical(grade, "exploratory")) {
    stop("unresolved concentration authority must project exploratory evidence")
  }
  list(concentration_status = status, evidence_grade = grade)
}

publication_warning_text <- function(
  data,
  concentration_status = "unresolved_no_preregistered_cutoff",
  evidence_grade = "exploratory"
) {
  labels <- unique(publication_warning_labels(data))
  paste(
    c(
      paste0(
        "EVIDENCE GATE: ", toupper(evidence_grade),
        " — concentration=", concentration_status
      ),
      sort(labels)
    ),
    collapse = " | "
  )
}

select_registered_component_interval <- function(
  clusters,
  conventional_conf_low,
  conventional_conf_high,
  conventional_p_value,
  wild_conf_low,
  wild_conf_high,
  wild_p_value,
  wild_inference_status
) {
  result <- data.frame(
    conf_low = rep(NA_real_, length(clusters)),
    conf_high = rep(NA_real_, length(clusters)),
    p_value = rep(NA_real_, length(clusters)),
    source = rep(
      "point_only_exploratory_no_registered_interval",
      length(clusters)
    ),
    stringsAsFactors = FALSE
  )
  wild <- clusters >= 20L & clusters < 30L
  bounded <- wild & wild_inference_status == "wild_bootstrap_bounded"
  unbounded <- wild & wild_inference_status == "wild_bootstrap_unbounded"
  if (any(wild & !(bounded | unbounded))) {
    stop("20-29 cluster reporting requires validated wild-bootstrap status")
  }
  if (any(bounded & (!is.finite(wild_conf_low) | !is.finite(wild_conf_high)))) {
    stop("bounded wild-bootstrap reporting interval is missing")
  }
  result$conf_low[bounded] <- wild_conf_low[bounded]
  result$conf_high[bounded] <- wild_conf_high[bounded]
  result$p_value[bounded] <- wild_p_value[bounded]
  result$source[bounded] <- "wild_cluster_bootstrap_9999"
  result$p_value[unbounded] <- wild_p_value[unbounded]
  result$source[unbounded] <- "wild_bootstrap_unbounded_point_only"
  conventional <- clusters >= 30L
  result$conf_low[conventional] <- conventional_conf_low[conventional]
  result$conf_high[conventional] <- conventional_conf_high[conventional]
  result$p_value[conventional] <- conventional_p_value[conventional]
  result$source[conventional] <- "conventional_cluster_t"
  result
}

threshold_publication_warning <- function(
  low_wild_status,
  high_wild_status,
  difference_wild_status,
  clusters
) {
  component_names <- c("LOW", "HIGH", "DIFFERENCE")
  statuses <- cbind(low_wild_status, high_wild_status, difference_wild_status)
  vapply(seq_len(nrow(statuses)), function(index) {
    component_status <- statuses[index, ]
    if (clusters[[index]] < 20L) {
      labels <- rep("POINT-ONLY/EXPLORATORY", length(component_status))
    } else if (clusters[[index]] < 30L) {
      labels <- rep("LOW-CLUSTER WILD BOOTSTRAP", length(component_status))
      labels[component_status == "wild_bootstrap_unbounded"] <-
        "POINT-ONLY/UNBOUNDED WILD INTERVAL"
    } else {
      labels <- rep("CONVENTIONAL CLUSTER-t", length(component_status))
    }
    paste(paste(component_names, labels, sep = ": "), collapse = " | ")
  }, character(1L))
}

build_threshold_table <- function(
  threshold,
  concentration_status = "unresolved_no_preregistered_cutoff"
) {
  key <- c("outcome_id", "horizon", "gad_version", "sample_version")
  required <- c(
    key, "regime", "estimate", "conf_low", "conf_high", "p_value",
    "wild_conf_low", "wild_conf_high", "wild_p_value",
    "wild_inference_status", "clusters", "inference_status",
    "difference_estimate", "difference_std_error", "difference_conf_low",
    "difference_conf_high", "difference_p_value",
    "difference_wild_conf_low", "difference_wild_conf_high",
    "difference_wild_p_value", "difference_wild_inference_status"
  )
  missing <- setdiff(required, names(threshold))
  if (length(missing)) {
    stop("threshold reporting inputs are missing validated fields: ", paste(missing, collapse = ", "))
  }
  low <- threshold[threshold$regime == "low", , drop = FALSE]
  high <- threshold[threshold$regime == "high", , drop = FALSE]
  low <- low[order(low$outcome_id, low$horizon, low$gad_version, low$sample_version), ]
  high <- high[order(high$outcome_id, high$horizon, high$gad_version, high$sample_version), ]
  paired_keys <- nrow(low) == nrow(high) && all(vapply(
    key,
    function(field) identical(as.character(low[[field]]), as.character(high[[field]])),
    logical(1L)
  ))
  if (!paired_keys) {
    stop("threshold reporting requires paired low/high validated rows")
  }
  paired <- grep("^difference_", names(low), value = TRUE)
  for (field in paired) {
    same <- (is.na(low[[field]]) & is.na(high[[field]])) |
      (!is.na(low[[field]]) & !is.na(high[[field]]) & low[[field]] == high[[field]])
    if (any(!same)) stop("threshold reporting paired contrast mismatch: ", field)
  }
  result <- low[key]
  result$low_estimate <- low$estimate
  low_interval <- select_registered_component_interval(
    low$clusters, low$conf_low, low$conf_high, low$p_value,
    low$wild_conf_low, low$wild_conf_high, low$wild_p_value,
    low$wild_inference_status
  )
  result$low_conf_low <- low_interval$conf_low
  result$low_conf_high <- low_interval$conf_high
  result$low_p_value <- low_interval$p_value
  result$low_conventional_conf_low <- low$conf_low
  result$low_conventional_conf_high <- low$conf_high
  result$low_conventional_p_value <- low$p_value
  result$registered_low_interval_source <- low_interval$source
  result$high_estimate <- high$estimate
  high_interval <- select_registered_component_interval(
    high$clusters, high$conf_low, high$conf_high, high$p_value,
    high$wild_conf_low, high$wild_conf_high, high$wild_p_value,
    high$wild_inference_status
  )
  result$high_conf_low <- high_interval$conf_low
  result$high_conf_high <- high_interval$conf_high
  result$high_p_value <- high_interval$p_value
  result$high_conventional_conf_low <- high$conf_low
  result$high_conventional_conf_high <- high$conf_high
  result$high_conventional_p_value <- high$p_value
  result$registered_high_interval_source <- high_interval$source
  for (field in paired) result[[field]] <- low[[field]]
  result$clusters <- low$clusters
  result$inference_status <- low$inference_status
  result$registered_difference_conf_low <- NA_real_
  result$registered_difference_conf_high <- NA_real_
  result$registered_difference_p_value <- NA_real_
  result$registered_interval_source <- "point_only_exploratory_no_registered_interval"
  wild <- low$clusters >= 20L & low$clusters < 30L
  bounded <- wild & low$difference_wild_inference_status == "wild_bootstrap_bounded"
  result$registered_difference_conf_low[bounded] <- low$difference_wild_conf_low[bounded]
  result$registered_difference_conf_high[bounded] <- low$difference_wild_conf_high[bounded]
  result$registered_difference_p_value[bounded] <- low$difference_wild_p_value[bounded]
  result$registered_interval_source[bounded] <- "wild_cluster_bootstrap_9999"
  unbounded <- wild & low$difference_wild_inference_status == "wild_bootstrap_unbounded"
  result$registered_difference_p_value[unbounded] <- low$difference_wild_p_value[unbounded]
  result$registered_interval_source[unbounded] <- "wild_bootstrap_unbounded_point_only"
  conventional <- low$clusters >= 30L
  result$registered_difference_conf_low[conventional] <- low$difference_conf_low[conventional]
  result$registered_difference_conf_high[conventional] <- low$difference_conf_high[conventional]
  result$registered_difference_p_value[conventional] <- low$difference_p_value[conventional]
  result$registered_interval_source[conventional] <- "conventional_cluster_t"
  result$publication_warning <- threshold_publication_warning(
    low$wild_inference_status,
    high$wild_inference_status,
    low$difference_wild_inference_status,
    low$clusters
  )
  result$concentration_status <- concentration_status
  rownames(result) <- NULL
  result
}

marginal_curve_interval_policy <- function(clusters, inference_status) {
  if (length(clusters) != 1L || length(inference_status) != 1L) {
    stop("marginal curve interval policy requires one registered cell")
  }
  if (clusters < 20L) {
    if (!identical(inference_status, "exploratory_lt20_clusters")) {
      stop("marginal curve exploratory status mismatch")
    }
    return("point_only_exploratory")
  }
  if (clusters < 30L) {
    if (identical(inference_status, "wild_bootstrap_unbounded")) {
      return("point_only_wild_bootstrap_unbounded")
    }
    if (!identical(inference_status, "wild_bootstrap_required")) {
      stop("marginal curve wild-bootstrap status mismatch")
    }
    return("point_only_no_registered_wild_curve")
  }
  if (!identical(inference_status, "cluster_robust")) {
    stop("marginal curve cluster-robust status mismatch")
  }
  "cluster_robust"
}

publication_tables <- function(inputs) {
  evidence_gate <- reporting_evidence_projection(inputs$evidence_policy)
  environmental <- c(
    "co2_tonnes_per_million_current_usd",
    "renewable_capacity_additions_mw_per_million",
    "energy_intensity_mj_per_ppp_gdp"
  )
  industrial <- c(
    "future_green_rca_entry_rate", "green_export_complexity",
    "green_export_share", "domestic_value_added_share",
    "foreign_value_added_dependence"
  )
  estimate_columns <- c(
    "estimator", "analysis_family", "outcome_id", "horizon", "term",
    "estimate", "std_error", "conf_low", "conf_high", "p_value",
    "wild_conf_low", "wild_conf_high", "wild_p_value", "wild_draws",
    "wild_seed", "n", "economies", "clusters", "inference_status",
    "ssc_config", "reference_distribution", "reference_df",
    "publication_conf_low", "publication_conf_high",
    "publication_interval_status"
  )
  lp_fe <- select_publication_intervals(inputs$lp_fe)
  lp_iv <- select_publication_intervals(inputs$lp_iv)
  estimates <- rbind(lp_fe[estimate_columns], lp_iv[estimate_columns])
  table_1 <- inputs$sample_cells[c(
    "analysis_family", "outcome_id", "horizon", "gad_version",
    "sample_version", "role", "candidate_rows", "n", "sample_loss",
    "economies", "clusters", "year_min", "year_max"
  )]
  table_2 <- bind_rows_fill(
    metric_rows(
      inputs$distributions,
      c("gad_version", "sample_version", "variable", "n"),
      c("mean", "std", "min", "p25", "p50", "p75", "max"),
      "distribution"
    ),
    metric_rows(
      inputs$correlations,
      c("gad_version", "sample_version", "variable_x", "variable_y", "n"),
      "correlation", "correlation"
    )
  )
  table_3 <- bind_rows_fill(
    metric_rows(
      inputs$first_stage_screen,
      c("analysis_family", "outcome_id", "horizon", "gad_version", "sample_version"),
      c(
        "instrument_rank", "cross_moment_rank", "partial_r2_gimc",
        "partial_r2_interaction", "effective_f_gimc", "effective_f_interaction"
      ),
      "first_stage"
    ),
    metric_rows(
      inputs$exposure_concentration,
      c("taxonomy_version", "share_version", "importer", "exposure_units"),
      c("hhi", "effective_units", "top1_share", "top5_share"),
      "exposure"
    )
  )
  table_4 <- estimates[
    estimates$analysis_family == "confirmatory" &
      estimates$outcome_id %in% environmental & estimates$horizon %in% 1:3,
    , drop = FALSE
  ]
  table_4$publication_warning <- publication_warning_labels(table_4)
  table_4$concentration_status <- as.character(
    inputs$evidence_policy$concentration$status
  )
  marginal_columns <- c(
    "estimator", "analysis_family", "outcome_id", "horizon", "gad_quantile",
    "gad_value", "estimate", "std_error", "conf_low", "conf_high", "p_value",
    "wild_conf_low", "wild_conf_high", "wild_p_value", "wild_draws",
    "wild_seed", "n", "economies", "clusters", "inference_status",
    "ssc_config", "reference_distribution", "reference_df",
    "publication_conf_low", "publication_conf_high",
    "publication_interval_status"
  )
  marginal_effects <- select_publication_intervals(inputs$marginal_effects)
  table_5 <- bind_rows_fill(
    transform(
      estimates[
        estimates$analysis_family == "confirmatory" &
          estimates$outcome_id %in% industrial & estimates$horizon %in% 3:8,
        , drop = FALSE
      ],
      row_type = "coefficient"
    ),
    transform(
      marginal_effects[
        marginal_effects$analysis_family == "confirmatory" &
          marginal_effects$outcome_id %in% industrial &
          marginal_effects$horizon %in% 3:8,
        marginal_columns, drop = FALSE
      ],
      row_type = "marginal_effect"
    )
  )
  table_5$concentration_status <- evidence_gate$concentration_status
  table_5$evidence_grade <- evidence_gate$evidence_grade
  table_5$publication_warning <- publication_warning_text(
    table_5,
    evidence_gate$concentration_status,
    evidence_gate$evidence_grade
  )
  table_6 <- build_threshold_table(
    inputs$threshold_estimates,
    as.character(inputs$evidence_policy$concentration$status)
  )
  table_7 <- estimates[
    estimates$analysis_family %in% c("bounded_controls", "vulnerability"),
    , drop = FALSE
  ]
  table_7$publication_warning <- publication_warning_labels(table_7)
  table_7$concentration_status <- as.character(
    inputs$evidence_policy$concentration$status
  )
  table_8 <- merge(
    inputs$weak_iv_sets,
    inputs$shift_share_summary,
    by = c(
      "run_id", "spec_id", "input_authority_hash", "git_commit",
      "renv_lock_sha256", "evidence_policy_sha256", "created_at_utc",
      "analysis_family", "outcome_id",
      "horizon", "gad_version", "sample_version"
    ),
    all = TRUE, suffixes = c("_ar", "_shock"), sort = TRUE
  )
  list(table_1, table_2, table_3, table_4, table_5, table_6, table_7, table_8)
}

figure_bundle <- function(data, plot, sources) {
  hash <- canonical_reporting_data_hash(data)
  attr(plot, "source_data_hash") <- hash
  list(data = data, plot = plot, sources = sources)
}

publication_figure_metadata <- function() {
  common <- list(
    role = "main",
    first_citation = "not_available_no_manuscript",
    output_formats = c("svg", "pdf", "png"),
    minimum_text_pt = 6.5
  )
  define <- function(
    figure_id, research_question, panels, analysis_unit,
    uncertainty_display, height, caption
  ) {
    c(
      list(
        figure_id = figure_id,
        research_question = research_question,
        panels = panels,
        analysis_unit = analysis_unit,
        uncertainty_display = uncertainty_display,
        target_canvas_mm = list(width = 183, height = height),
        caption = caption,
        source_files = character()
      ),
      common
    )
  }
  list(
    define(
      "figure_1",
      "How do the registered import, absorptive-debt, instrument, and control distributions vary across analysis definitions?",
      "Eight metric-specific small multiples compare all seven registered GAD/sample combinations.",
      "Economy-year observations in the registered 2000–2022 analysis samples.",
      "Points show medians; horizontal intervals show the 25th–75th percentiles.",
      160,
      paste(
        "Figure 1. Registered analysis distributions.",
        "The eight panels report medians and interquartile ranges for GIMC, lagged GAD,",
        "the shift-share instrument and its GAD interaction, and the registered controls",
        "across all GAD and sample definitions. Values remain in their native units and",
        "each metric uses its own horizontal scale, so distances must not be compared",
        "across panels. The analytical unit is the economy-year over 2000–2022; the",
        "figure is descriptive rather than causal."
      )
    ),
    define(
      "figure_2",
      "How do environmental and industrial outcomes evolve descriptively after the baseline year?",
      "Eight outcome-specific paths are arranged as environmental and industrial small multiples.",
      "Mean economy-year outcome change at horizons 0–8 in the registered complete-case samples.",
      "No inferential interval is attached; every panel is explicitly descriptive and noncausal.",
      150,
      paste(
        "Figure 2. Descriptive outcome paths.",
        "Panels trace the mean change from baseline for three environmental and five",
        "industrial outcomes over their registered horizons. Each outcome remains in its",
        "native unit and uses an independent vertical scale; the horizontal zero line marks",
        "no mean change, and the final observed value is labelled directly. Samples cover",
        "economy-year observations from 2000–2022.",
        "These paths are descriptive and must not be interpreted causally."
      )
    ),
    define(
      "figure_3",
      "Do imported green capabilities improve environmental outcomes over short horizons?",
      "Three environmental outcome panels show LP-IV estimates at horizons 1–3.",
      "Economy-year LP-IV cells in the registered complete-case sample.",
      "Points are LP-IV estimates; available registered 95% intervals are shown, while unbounded wild-bootstrap cells remain point-only.",
      105,
      paste(
        "Figure 3. Short-horizon environmental LP-IV estimates.",
        "Each panel reports the GIMC coefficient at horizons 1–3 for one environmental",
        "outcome. Points are estimates and horizontal bars are the registered 95% intervals selected",
        "by the frozen cluster rule. Open point-only symbols identify cells whose registered",
        "wild-bootstrap interval is unbounded; no conventional interval is substituted.",
        "The zero line marks no effect. Evidence remains exploratory because the",
        "concentration cutoff was not preregistered."
      )
    ),
    define(
      "figure_4",
      "How does accumulated GAD change the medium-run industrial return to imported green capabilities?",
      "Five outcome panels compare the interaction coefficient with marginal effects at the 25th, 50th, and 75th GAD percentiles over horizons 3–8.",
      "Economy-year LP-IV cells in the registered complete-case samples.",
      "Points show estimates; horizontal bars show registered 95% cluster-t intervals.",
      175,
      paste(
        "Figure 4. Industrial interactions and GAD-dependent marginal effects.",
        "For each industrial outcome and horizon 3–8, the figure retains the GIMC×GAD",
        "interaction and the marginal GIMC effects evaluated at the 25th, 50th, and 75th",
        "percentiles of the registered GAD distribution. Points are estimates and horizontal bars are",
        "registered 95% cluster-t intervals; the zero line marks no effect. All 120 plotted",
        "cells are retained. Evidence remains exploratory because the concentration cutoff",
        "was not preregistered."
      )
    ),
    define(
      "figure_5",
      "Where does the marginal effect of imported green capability cross zero over the observed GAD range?",
      "A continuous horizon-5 marginal-effect curve for green export complexity marks the frozen GAD threshold.",
      "Registered economy-year LP-IV cell evaluated over the observed 1st–99th percentile GAD range.",
      "The line is the marginal estimate and the band is its registered 95% cluster-t interval.",
      105,
      paste(
        "Figure 5. Marginal import effect across the observed GAD range.",
        "The line reports the horizon-5 marginal effect of GIMC on green export complexity",
        "and the shaded band is the registered 95% cluster-t interval computed from the",
        "full coefficient covariance matrix. The horizontal line marks zero; the vertical",
        "line marks the frozen threshold q = 1.48. The GAD axis spans the registered",
        "1st–99th percentile range. Evidence remains exploratory because the concentration",
        "cutoff was not preregistered."
      )
    ),
    define(
      "figure_6",
      "How vulnerable are green imports and renewable deployment following negative partner-supply shocks?",
      "Two outcome panels show vulnerability LP-IV paths at horizons 1–3.",
      "Economy-year LP-IV cells in the registered negative-shock complete-case sample.",
      "Points are estimates; available registered 95% intervals are shown and low-cluster cells remain explicitly point-only.",
      105,
      paste(
        "Figure 6. Negative-shock vulnerability paths.",
        "Panels report horizon 1–3 LP-IV estimates for asinh weighted green imports and",
        "renewable-capacity additions after negative partner-supply shocks. Horizontal bars are the",
        "registered 95% intervals when admissible. Open symbols identify",
        "low-cluster point-only cells; no interval is imputed. The zero line marks no",
        "effect. Evidence remains exploratory."
      )
    ),
    define(
      "figure_7",
      "Which partner-product-year shocks dominate the shift-share instrument, and how concentrated are the weights?",
      c(
        "Panel A ranks all 25 retained absolute Rotemberg contributors.",
        "Panel B reports mean HHI, top-1, and top-5 absolute-weight concentration."
      ),
      "Partner-product-year shocks aggregated across registered outcome-horizon cells.",
      "Descriptive weight diagnostics; no confidence interval is defined.",
      180,
      paste(
        "Figure 7. Rotemberg contributors and concentration diagnostics.",
        "Panel A ranks all 25 retained partner-product-year shocks by aggregated absolute",
        "Rotemberg contribution. Panel B reports the mean absolute-weight HHI and top-1",
        "and top-5 shares across registered cells. Blue points highlight the five largest",
        "contributors, and values are labelled directly. Country codes, HS6 products, and",
        "years identify shocks. These diagnostics disclose concentration but do not impose an",
        "unregistered pass/fail cutoff; the evidence gate therefore remains exploratory."
      )
    )
  )
}

publication_source_files <- function(sources) {
  vapply(sources, function(source) {
    if (identical(source, "threshold_registry")) {
      return("registries/threshold_registry_v1.json")
    }
    path <- reporting_table_paths[[source]]
    if (is.null(path)) stop("unknown figure source: ", source)
    unname(path)
  }, character(1L), USE.NAMES = FALSE)
}

publication_theme <- function() {
  ggplot2::theme_minimal(base_family = "Arial", base_size = 7.5) +
    ggplot2::theme(
      text = ggplot2::element_text(colour = "#17213A"),
      plot.background = ggplot2::element_rect(fill = "white", colour = NA),
      panel.background = ggplot2::element_rect(fill = "white", colour = NA),
      panel.grid.major = ggplot2::element_blank(),
      panel.grid.major.x = ggplot2::element_blank(),
      panel.grid.major.y = ggplot2::element_line(
        colour = "#E8EBF0", linewidth = 0.24
      ),
      panel.grid.minor = ggplot2::element_blank(),
      plot.title = ggplot2::element_text(
        family = "Arial", face = "bold", size = 8.8, hjust = 0,
        margin = ggplot2::margin(b = 1.8)
      ),
      plot.subtitle = ggplot2::element_text(
        family = "Arial", size = 7.2, colour = "#596270",
        margin = ggplot2::margin(b = 4.5)
      ),
      strip.text = ggplot2::element_text(
        family = "Arial", face = "bold", size = 8.5, hjust = 0,
        colour = "#17213A",
        margin = ggplot2::margin(t = 2.5, r = 1.5, b = 3.5, l = 0)
      ),
      strip.background = ggplot2::element_blank(),
      axis.title = ggplot2::element_text(family = "Arial", size = 7.5),
      axis.text = ggplot2::element_text(family = "Arial", size = 6.7),
      axis.ticks.x = ggplot2::element_line(colour = "#9AA3AF", linewidth = 0.3),
      axis.ticks.length.x = grid::unit(1.1, "mm"),
      axis.line.x = ggplot2::element_line(colour = "#9AA3AF", linewidth = 0.3),
      legend.position = "bottom",
      legend.title = ggplot2::element_text(
        family = "Arial", face = "bold", size = 7
      ),
      legend.text = ggplot2::element_text(family = "Arial", size = 6.7),
      legend.key.height = grid::unit(3.5, "mm"),
      legend.key.width = grid::unit(5, "mm"),
      legend.spacing.x = grid::unit(1.2, "mm"),
      plot.caption = ggplot2::element_text(
        family = "Arial", size = 6.5, colour = "#6B7280", hjust = 0,
        margin = ggplot2::margin(t = 3)
      ),
      plot.margin = ggplot2::margin(8.5, 9, 8.5, 9)
    )
}

publication_outcome_label <- function(values) {
  labels <- c(
    co2_tonnes_per_million_current_usd = "CO₂ intensity",
    renewable_capacity_additions_mw_per_million = "Renewable capacity\nadditions",
    energy_intensity_mj_per_ppp_gdp = "Energy intensity",
    future_green_rca_entry_rate = "Future green RCA\nentry",
    green_export_complexity = "Green export\ncomplexity",
    green_export_share = "Green export share",
    domestic_value_added_share = "Domestic\nvalue-added share",
    foreign_value_added_dependence = "Foreign value-added\ndependence",
    asinh_weighted_green_imports = "Weighted green imports\n(asinh)"
  )
  result <- unname(labels[as.character(values)])
  if (any(is.na(result))) {
    stop("missing reader-facing outcome label: ", paste(unique(values[is.na(result)]), collapse = ", "))
  }
  result
}

publication_metric_label <- function(values) {
  labels <- c(
    gimc_p01_p99 = "Green import\ncapability (GIMC)",
    gad_lag_p01_p99 = "Lagged green\nabsorptive debt (GAD)",
    Z_p01_p99 = "Shift-share\ninstrument",
    Z_GAD_p01_p99 = "Instrument × GAD",
    renewable_energy_consumption_share_analysis_p01_p99 = "Renewable energy\nshare",
    trade_openness_percent_gdp_analysis_p01_p99 = "Trade openness",
    industry_value_added_share_analysis_p01_p99 = "Industry\nvalue-added share",
    gdp_per_capita_current_usd_analysis_p01_p99 = "GDP per capita"
  )
  result <- unname(labels[as.character(values)])
  if (any(is.na(result))) {
    stop("missing reader-facing metric label: ", paste(unique(values[is.na(result)]), collapse = ", "))
  }
  result
}

publication_specification_label <- function(gad_version, sample_version) {
  gad <- c(
    gad_core = "Core GAD",
    gad_no_supp = "No supplements",
    gad_no_gfvad = "No FVA component"
  )
  sample <- c(
    core_complete_case = "Complete case",
    core_bounded_controls = "Bounded controls",
    core_complete_case_negative_shock = "Negative shock"
  )
  gad_label <- unname(gad[as.character(gad_version)])
  sample_label <- unname(sample[as.character(sample_version)])
  if (any(is.na(gad_label)) || any(is.na(sample_label))) {
    stop("missing reader-facing specification label")
  }
  paste(gad_label, sample_label, sep = " · ")
}

publication_interval_label <- function(status, point_only_label) {
  available <- status %in% c("cluster_robust", "wild_bootstrap")
  ifelse(available, "Registered 95% interval", point_only_label)
}

publication_figures <- function(inputs) {
  theme <- publication_theme()
  colours <- list(
    navy = "#17213A", blue = "#0072B2", green = "#009E73",
    orange = "#D55E00", purple = "#7B3294", grey = "#9AA3AF"
  )
  short_axis_labels <- scales::label_number(
    scale_cut = scales::cut_short_scale(), accuracy = NULL
  )
  evidence_gate <- reporting_evidence_projection(inputs$evidence_policy)
  distributions <- inputs$distributions
  distributions$metric_label <- publication_metric_label(distributions$variable)
  distributions$specification_label <- publication_specification_label(
    distributions$gad_version, distributions$sample_version
  )
  metric_order <- publication_metric_label(c(
    "gimc_p01_p99", "gad_lag_p01_p99", "Z_p01_p99", "Z_GAD_p01_p99",
    "renewable_energy_consumption_share_analysis_p01_p99",
    "trade_openness_percent_gdp_analysis_p01_p99",
    "industry_value_added_share_analysis_p01_p99",
    "gdp_per_capita_current_usd_analysis_p01_p99"
  ))
  distributions$metric_label <- factor(
    distributions$metric_label, levels = metric_order
  )
  metric_panel_labels <- stats::setNames(
    paste0(LETTERS[seq_along(metric_order)], "  ", metric_order),
    as.character(metric_order)
  )
  distributions$panel_label <- factor(
    unname(metric_panel_labels[as.character(distributions$metric_label)]),
    levels = unname(metric_panel_labels)
  )
  f1 <- figure_bundle(
    distributions,
    ggplot2::ggplot(
      distributions,
      ggplot2::aes(
        y = specification_label, x = p50, xmin = p25, xmax = p75
      )
    ) +
      ggplot2::geom_vline(
        xintercept = 0, colour = "#D8DDE4", linewidth = 0.3
      ) +
      ggplot2::geom_segment(
        ggplot2::aes(x = p25, xend = p75, yend = specification_label),
        colour = "#AEB6C1", linewidth = 0.75, lineend = "round"
      ) +
      ggplot2::geom_point(
        shape = 21, size = 1.75, stroke = 0.35,
        colour = "white", fill = colours$blue
      ) +
      ggplot2::facet_wrap(~panel_label, scales = "free_x", ncol = 4) +
      ggplot2::scale_x_continuous(
        labels = scales::label_number(
          scale_cut = scales::cut_short_scale(), accuracy = NULL
        ),
        expand = ggplot2::expansion(mult = c(0.04, 0.06))
      ) +
      ggplot2::labs(
        title = "Registered distributions",
        subtitle = "Median • interquartile range • native units",
        x = "Value", y = NULL
      ) + theme +
      ggplot2::theme(
        panel.spacing = grid::unit(2.4, "mm"),
        axis.text.y = ggplot2::element_text(colour = "#525B67")
      ),
    c("distributions")
  )
  descriptive <- inputs$descriptive_paths
  descriptive$outcome_label <- publication_outcome_label(descriptive$outcome_id)
  descriptive$domain_label <- ifelse(
    descriptive$outcome_id %in% c(
      "co2_tonnes_per_million_current_usd",
      "renewable_capacity_additions_mw_per_million",
      "energy_intensity_mj_per_ppp_gdp"
    ),
    "Environmental", "Industrial"
  )
  descriptive$outcome_label <- factor(
    descriptive$outcome_label,
    levels = publication_outcome_label(c(
      "co2_tonnes_per_million_current_usd",
      "renewable_capacity_additions_mw_per_million",
      "energy_intensity_mj_per_ppp_gdp",
      "future_green_rca_entry_rate", "green_export_complexity",
      "green_export_share", "domestic_value_added_share",
      "foreign_value_added_dependence"
    ))
  )
  descriptive_levels <- levels(descriptive$outcome_label)
  descriptive_panel_labels <- stats::setNames(
    paste0(LETTERS[seq_along(descriptive_levels)], "  ", descriptive_levels),
    descriptive_levels
  )
  descriptive$panel_label <- factor(
    unname(descriptive_panel_labels[as.character(descriptive$outcome_label)]),
    levels = unname(descriptive_panel_labels)
  )
  last_horizon <- ave(
    descriptive$horizon,
    descriptive$outcome_id,
    FUN = max
  )
  descriptive$endpoint_label <- ifelse(
    descriptive$horizon == last_horizon,
    trimws(formatC(
      descriptive$mean_delta_outcome, format = "fg", digits = 3L
    )),
    NA_character_
  )
  f2 <- figure_bundle(
    descriptive,
    ggplot2::ggplot(
      descriptive,
      ggplot2::aes(
        horizon, mean_delta_outcome, colour = domain_label,
        group = interaction(outcome_id, gad_version, sample_version)
      )
    ) +
      ggplot2::geom_hline(
        yintercept = 0, colour = "#B8C0CA", linewidth = 0.35
      ) +
      ggplot2::geom_line(linewidth = 0.7, lineend = "round") +
      ggplot2::geom_point(size = 1.1) +
      ggplot2::geom_text(
        data = descriptive[!is.na(descriptive$endpoint_label), , drop = FALSE],
        ggplot2::aes(label = endpoint_label),
        hjust = 1.15, vjust = -0.35, family = "Arial",
        fontface = "bold", size = 2.35, show.legend = FALSE
      ) +
      ggplot2::facet_wrap(~panel_label, scales = "free_y", ncol = 4) +
      ggplot2::scale_colour_manual(
        values = c(Environmental = colours$blue, Industrial = colours$orange),
        guide = "none"
      ) +
      ggplot2::scale_x_continuous(
        breaks = seq(0, 8, 2),
        expand = ggplot2::expansion(mult = c(0.03, 0.08))
      ) +
      ggplot2::labs(
        title = "Descriptive outcome paths",
        subtitle = "Mean change from baseline • descriptive, not causal",
        x = "Horizon (years)", y = "Mean change"
      ) + theme +
      ggplot2::theme(panel.spacing = grid::unit(2.8, "mm")),
    c("descriptive_paths")
  )
  environmental <- c(
    "co2_tonnes_per_million_current_usd",
    "renewable_capacity_additions_mw_per_million",
    "energy_intensity_mj_per_ppp_gdp"
  )
  lp_iv <- select_publication_intervals(inputs$lp_iv)
  marginal_effects <- select_publication_intervals(inputs$marginal_effects)
  lp_iv_h1 <- lp_iv[
    lp_iv$analysis_family == "confirmatory" &
      lp_iv$outcome_id %in% environmental & lp_iv$term == "gimc_a",
    , drop = FALSE
  ]
  lp_iv_h1$publication_warning <- publication_warning_labels(lp_iv_h1)
  lp_iv_h1$concentration_status <- as.character(
    inputs$evidence_policy$concentration$status
  )
  f3_warning <- publication_warning_text(
    lp_iv_h1, lp_iv_h1$concentration_status[[1L]]
  )
  lp_iv_h1$outcome_label <- publication_outcome_label(lp_iv_h1$outcome_id)
  environmental_levels <- publication_outcome_label(environmental)
  environmental_panel_labels <- stats::setNames(
    paste0(LETTERS[seq_along(environmental_levels)], "  ", environmental_levels),
    environmental_levels
  )
  lp_iv_h1$panel_label <- factor(
    unname(environmental_panel_labels[lp_iv_h1$outcome_label]),
    levels = unname(environmental_panel_labels)
  )
  lp_iv_h1$interval_label <- publication_interval_label(
    lp_iv_h1$publication_interval_status, "Point only — unbounded"
  )
  f3 <- figure_bundle(
    lp_iv_h1,
    ggplot2::ggplot(
      lp_iv_h1,
      ggplot2::aes(
        estimate, horizon, xmin = publication_conf_low,
        xmax = publication_conf_high,
        colour = interval_label, shape = interval_label
      )
    ) +
      ggplot2::geom_vline(
        xintercept = 0, colour = "#AEB6C1", linewidth = 0.4
      ) +
      ggplot2::geom_errorbar(
        orientation = "y", width = 0.18, linewidth = 0.58, na.rm = TRUE
      ) +
      ggplot2::geom_point(size = 2.05, stroke = 0.7) +
      ggplot2::facet_wrap(~panel_label, scales = "free_x", nrow = 1) +
      ggplot2::scale_colour_manual(values = c(
        "Registered 95% interval" = colours$blue,
        "Point only — unbounded" = colours$orange
      )) +
      ggplot2::scale_shape_manual(values = c(
        "Registered 95% interval" = 16,
        "Point only — unbounded" = 1
      )) +
      ggplot2::scale_y_continuous(
        breaks = 1:3, expand = ggplot2::expansion(mult = c(0.16, 0.16))
      ) +
      ggplot2::scale_x_continuous(labels = short_axis_labels) +
      ggplot2::labs(
        title = "Short-horizon environmental effects",
        subtitle = "Exploratory evidence • registered inference retained",
        x = "LP-IV estimate", y = "Horizon (years)",
        colour = NULL, shape = NULL
      ) + theme +
      ggplot2::theme(
        legend.position = "top",
        legend.justification = "left",
        legend.box.just = "left",
        panel.spacing = grid::unit(4, "mm")
      ),
    c("lp_iv")
  )
  industrial_outcomes <- c(
    "future_green_rca_entry_rate", "green_export_complexity",
    "green_export_share", "domestic_value_added_share",
    "foreign_value_added_dependence"
  )
  industrial <- lp_iv[
    lp_iv$analysis_family == "confirmatory" &
      lp_iv$outcome_id %in% industrial_outcomes &
      lp_iv$horizon %in% 3:8 & lp_iv$term == "gimc_gad_a",
    , drop = FALSE
  ]
  industrial_marginal <- marginal_effects[
      marginal_effects$estimator == "lp_iv" &
      marginal_effects$analysis_family == "confirmatory" &
      marginal_effects$outcome_id %in% industrial_outcomes &
      marginal_effects$horizon %in% 3:8,
    , drop = FALSE
  ]
  industrial$line_group <- paste(industrial$outcome_id, "interaction", sep = "|")
  industrial_marginal$line_group <- paste(
    industrial_marginal$outcome_id,
    "marginal",
    industrial_marginal$gad_quantile,
    sep = "|"
  )
  f4_data <- bind_rows_fill(
    transform(industrial, series = "interaction", gad_quantile = NA_real_),
    transform(industrial_marginal, series = "marginal")
  )
  f4_data$concentration_status <- evidence_gate$concentration_status
  f4_data$evidence_grade <- evidence_gate$evidence_grade
  f4_warning <- publication_warning_text(
    f4_data,
    evidence_gate$concentration_status,
    evidence_gate$evidence_grade
  )
  f4_data$publication_warning <- f4_warning
  f4_data$outcome_label <- publication_outcome_label(f4_data$outcome_id)
  industrial_levels <- publication_outcome_label(industrial_outcomes)
  industrial_panel_labels <- stats::setNames(
    paste0(LETTERS[seq_along(industrial_levels)], "  ", industrial_levels),
    industrial_levels
  )
  f4_data$panel_label <- factor(
    unname(industrial_panel_labels[f4_data$outcome_label]),
    levels = unname(industrial_panel_labels)
  )
  f4_data$series_label <- ifelse(
    f4_data$series == "interaction",
    "Interaction",
    paste0("GAD Q", sprintf("%02d", round(100 * f4_data$gad_quantile)))
  )
  f4_data$series_label <- factor(
    f4_data$series_label,
    levels = c("Interaction", "GAD Q25", "GAD Q50", "GAD Q75")
  )
  f4_position <- ggplot2::position_dodge(width = 0.56, orientation = "y")
  f4 <- figure_bundle(
    f4_data,
    ggplot2::ggplot(
      f4_data,
      ggplot2::aes(
        estimate, horizon, xmin = publication_conf_low,
        xmax = publication_conf_high, colour = series_label,
        shape = series_label, group = series_label
      )
    ) +
      ggplot2::geom_vline(
        xintercept = 0, colour = "#AEB6C1", linewidth = 0.4
      ) +
      ggplot2::geom_errorbar(
        orientation = "y", width = 0.12, linewidth = 0.4, alpha = 0.68,
        position = f4_position
      ) +
      ggplot2::geom_point(
        size = 1.55, stroke = 0.5, position = f4_position
      ) +
      ggplot2::facet_wrap(~panel_label, scales = "free_x", ncol = 2) +
      ggplot2::scale_colour_manual(values = c(
        "Interaction" = "#756184",
        "GAD Q25" = "#7FB9D6",
        "GAD Q50" = colours$blue,
        "GAD Q75" = colours$orange
      )) +
      ggplot2::scale_shape_manual(values = c(
        "Interaction" = 15, "GAD Q25" = 16,
        "GAD Q50" = 17, "GAD Q75" = 18
      )) +
      ggplot2::scale_y_continuous(
        breaks = 3:8, expand = ggplot2::expansion(mult = c(0.09, 0.09))
      ) +
      ggplot2::scale_x_continuous(labels = short_axis_labels) +
      ggplot2::labs(
        title = "Industrial returns across the GAD distribution",
        subtitle = "Exploratory evidence • point estimates and registered 95% intervals",
        x = "LP-IV estimate", y = "Horizon (years)",
        colour = NULL, shape = NULL
      ) + theme +
      ggplot2::theme(
        legend.position = "top",
        legend.justification = "left",
        legend.box.just = "left",
        legend.key.width = grid::unit(4, "mm"),
        panel.spacing = grid::unit(4, "mm")
      ),
    c("lp_iv", "marginal_effects")
  )
  selected <- lp_iv[
    lp_iv$analysis_family == "confirmatory" &
      lp_iv$outcome_id == "green_export_complexity" &
      lp_iv$horizon == 5L,
    , drop = FALSE
  ]
  selected_covariance <- inputs$model_covariance[
    inputs$model_covariance$estimator == "lp_iv" &
      inputs$model_covariance$analysis_family == "confirmatory" &
      inputs$model_covariance$outcome_id == "green_export_complexity" &
      inputs$model_covariance$horizon == 5L,
    , drop = FALSE
  ]
  gad_range <- inputs$distributions[
    inputs$distributions$variable == "gad_lag_p01_p99" &
      inputs$distributions$gad_version == "gad_no_supp" &
      inputs$distributions$sample_version == "core_complete_case",
    , drop = FALSE
  ]
  gad <- seq(gad_range$min[[1L]], gad_range$max[[1L]], length.out = 101L)
  beta <- selected$estimate[selected$term == "gimc_a"]
  theta <- selected$estimate[selected$term == "gimc_gad_a"]
  covariance <- function(i, j) selected_covariance$covariance[
    selected_covariance$term_i == i & selected_covariance$term_j == j
  ][[1L]]
  variance <- covariance("gimc_a", "gimc_a") +
    gad^2 * covariance("gimc_gad_a", "gimc_gad_a") +
    2 * gad * covariance("gimc_a", "gimc_gad_a")
  cell_clusters <- unique(selected$clusters)
  cell_status <- unique(selected$inference_status)
  if (length(cell_clusters) != 1L || length(cell_status) != 1L) {
    stop("marginal curve coefficient terms do not share one inference cell")
  }
  interval_policy <- marginal_curve_interval_policy(cell_clusters, cell_status)
  curve <- data.frame(
    gad_value = gad, estimate = beta + gad * theta,
    std_error = NA_real_, conf_low = NA_real_, conf_high = NA_real_,
    q = as.numeric(inputs$threshold_registry$q),
    publication_interval_status = interval_policy
  )
  f5_warning <- publication_warning_text(
    selected,
    evidence_gate$concentration_status,
    evidence_gate$evidence_grade
  )
  curve$concentration_status <- evidence_gate$concentration_status
  curve$evidence_grade <- evidence_gate$evidence_grade
  curve$publication_warning <- f5_warning
  if (identical(interval_policy, "cluster_robust")) {
    curve$std_error <- sqrt(pmax(variance, 0))
    reference_df <- unique(selected$reference_df)
    if (length(reference_df) != 1L || !is.finite(reference_df) || reference_df <= 0) {
      stop("marginal curve cluster t reference df is invalid")
    }
    critical <- stats::qt(0.975, df = reference_df)
    curve$conf_low <- curve$estimate - critical * curve$std_error
    curve$conf_high <- curve$estimate + critical * curve$std_error
  }
  f5_plot <- ggplot2::ggplot(
    curve, ggplot2::aes(gad_value, estimate, ymin = conf_low, ymax = conf_high)
  )
  if (any(is.finite(curve$conf_low) & is.finite(curve$conf_high))) {
    f5_plot <- f5_plot + ggplot2::geom_ribbon(
      fill = colours$blue, alpha = 0.1, colour = NA
    )
  }
  curve_range <- range(c(curve$conf_low, curve$conf_high), na.rm = TRUE)
  q_y <- curve_range[[2L]] - 0.06 * diff(curve_range)
  f5_plot <- f5_plot +
    ggplot2::geom_hline(yintercept = 0, colour = "#9AA3AF", linewidth = 0.45) +
    ggplot2::geom_line(
      colour = colours$blue, linewidth = 0.78, lineend = "round"
    ) +
    ggplot2::geom_vline(
      ggplot2::aes(xintercept = q),
      colour = colours$orange, linetype = "22", linewidth = 0.55
    ) +
    ggplot2::annotate(
      "text", x = unique(curve$q), y = q_y,
      label = sprintf("Frozen threshold\nq = %.2f", unique(curve$q)),
      hjust = -0.08, vjust = 1, family = "Arial", size = 2.35,
      colour = colours$orange
    ) +
    ggplot2::labs(
      title = "Green export complexity at horizon 5",
      subtitle = "Exploratory evidence • marginal effect over the observed GAD range",
      x = "Lagged GAD (standardized index)",
      y = "Marginal GIMC effect",
      caption = "Blue band: registered 95% cluster-t interval."
    ) + theme +
    ggplot2::theme(panel.grid.major.y = ggplot2::element_line(
      colour = "#E8EBF0", linewidth = 0.24
    ))
  f5 <- figure_bundle(
    curve,
    f5_plot,
    c("lp_iv", "model_covariance", "distributions", "threshold_registry")
  )
  vulnerability <- lp_iv[
    lp_iv$analysis_family == "vulnerability" & lp_iv$term == "gimc_a",
    , drop = FALSE
  ]
  vulnerability$publication_warning <- publication_warning_labels(vulnerability)
  vulnerability$concentration_status <- as.character(
    inputs$evidence_policy$concentration$status
  )
  f6_warning <- publication_warning_text(
    vulnerability, vulnerability$concentration_status[[1L]]
  )
  vulnerability$outcome_label <- publication_outcome_label(vulnerability$outcome_id)
  vulnerability_levels <- unique(vulnerability$outcome_label)
  vulnerability_panel_labels <- stats::setNames(
    paste0(LETTERS[seq_along(vulnerability_levels)], "  ", vulnerability_levels),
    vulnerability_levels
  )
  vulnerability$panel_label <- factor(
    unname(vulnerability_panel_labels[vulnerability$outcome_label]),
    levels = unname(vulnerability_panel_labels)
  )
  vulnerability$interval_label <- publication_interval_label(
    vulnerability$publication_interval_status, "Point only — low clusters"
  )
  f6 <- figure_bundle(
    vulnerability,
    ggplot2::ggplot(
      vulnerability,
      ggplot2::aes(
        estimate, horizon, xmin = publication_conf_low,
        xmax = publication_conf_high,
        colour = interval_label, shape = interval_label
      )
    ) +
      ggplot2::geom_vline(
        xintercept = 0, colour = "#AEB6C1", linewidth = 0.4
      ) +
      ggplot2::geom_errorbar(
        orientation = "y", width = 0.18, linewidth = 0.58, na.rm = TRUE
      ) +
      ggplot2::geom_point(size = 2.05, stroke = 0.7) +
      ggplot2::facet_wrap(~panel_label, scales = "free_x", nrow = 1) +
      ggplot2::scale_colour_manual(values = c(
        "Registered 95% interval" = colours$orange,
        "Point only — low clusters" = "#687383"
      )) +
      ggplot2::scale_shape_manual(values = c(
        "Registered 95% interval" = 16,
        "Point only — low clusters" = 1
      )) +
      ggplot2::scale_y_continuous(
        breaks = sort(unique(vulnerability$horizon)),
        expand = ggplot2::expansion(mult = c(0.16, 0.16))
      ) +
      ggplot2::scale_x_continuous(labels = short_axis_labels) +
      ggplot2::labs(
        title = "Negative supply-shock vulnerability",
        subtitle = "Exploratory evidence • point-only cells retained",
        x = "LP-IV estimate", y = "Horizon (years)",
        colour = NULL, shape = NULL
      ) + theme +
      ggplot2::theme(
        legend.position = "top",
        legend.justification = "left",
        legend.box.just = "left",
        panel.spacing = grid::unit(5, "mm")
      ),
    c("lp_iv")
  )
  weights <- aggregate(
    absolute_weight ~ shock_id + exporter + hs6,
    inputs$shift_share_weights, sum
  )
  weights <- head(weights[order(-weights$absolute_weight, weights$shock_id), ], 25L)
  contributor_rows <- data.frame(
    section = "top_contributors", label = weights$shock_id,
    value = weights$absolute_weight,
    panel_label = "A  Top absolute contributors",
    display_label = gsub("[|]", " · ", weights$shock_id),
    emphasis = ifelse(seq_len(nrow(weights)) <= 5L, "Top five", "Other"),
    stringsAsFactors = FALSE
  )
  concentration_metrics <- c(
    "hhi_absolute", "top1_absolute_share", "top5_absolute_share"
  )
  concentration_rows <- data.frame(
    section = "concentration", label = concentration_metrics,
    value = vapply(
      concentration_metrics,
      function(metric) mean(inputs$shift_share_summary[[metric]], na.rm = TRUE),
      numeric(1L)
    ),
    panel_label = "B  Mean concentration diagnostics",
    display_label = c("Absolute-weight HHI", "Top-1 absolute share", "Top-5 absolute share"),
    emphasis = "Diagnostic",
    stringsAsFactors = FALSE
  )
  f7_data <- rbind(contributor_rows, concentration_rows)
  f7_data$value_label <- scales::label_number(accuracy = 0.001)(f7_data$value)
  f7_data$panel_label <- factor(
    f7_data$panel_label,
    levels = c(
      "A  Top absolute contributors",
      "B  Mean concentration diagnostics"
    )
  )
  f7 <- figure_bundle(
    f7_data,
    ggplot2::ggplot(
      f7_data,
      ggplot2::aes(x = value, y = stats::reorder(display_label, value))
    ) +
      ggplot2::geom_segment(
        data = f7_data[f7_data$section == "top_contributors", , drop = FALSE],
        ggplot2::aes(x = 0, xend = value, yend = stats::reorder(display_label, value)),
        colour = "#D2D7DE", linewidth = 0.5, lineend = "round"
      ) +
      ggplot2::geom_point(
        data = f7_data[f7_data$section == "top_contributors", , drop = FALSE],
        ggplot2::aes(colour = emphasis),
        shape = 16, size = 1.75
      ) +
      ggplot2::geom_col(
        data = f7_data[f7_data$section == "concentration", , drop = FALSE],
        fill = colours$orange, alpha = 0.18, width = 0.42
      ) +
      ggplot2::geom_point(
        data = f7_data[f7_data$section == "concentration", , drop = FALSE],
        colour = colours$orange, shape = 16, size = 1.9
      ) +
      ggplot2::geom_text(
        ggplot2::aes(label = value_label, colour = emphasis),
        hjust = -0.12, family = "Arial", size = 2.35,
        show.legend = FALSE
      ) +
      ggplot2::facet_wrap(
        ggplot2::vars(panel_label), ncol = 1L,
        scales = "free", space = "free_y"
      ) +
      ggplot2::scale_colour_manual(values = c(
        "Top five" = colours$blue,
        "Other" = "#7D8794",
        "Diagnostic" = colours$orange
      ), guide = "none") +
      ggplot2::scale_x_continuous(
        labels = scales::label_number(accuracy = 0.001),
        expand = ggplot2::expansion(mult = c(0, 0.16))
      ) +
      ggplot2::labs(
        title = "Instrument contribution and concentration",
        subtitle = "Descriptive diagnostics • no preregistered cutoff",
        x = "Diagnostic value (panel-specific scale)", y = NULL
      ) + theme +
      ggplot2::theme(
        legend.position = "none",
        panel.spacing = grid::unit(4.5, "mm"),
        axis.text.y = ggplot2::element_text(colour = "#525B67")
      ),
    c("shift_share_weights", "shift_share_summary")
  )
  figures <- list(f1, f2, f3, f4, f5, f6, f7)
  metadata <- publication_figure_metadata()
  for (index in seq_along(figures)) {
    metadata[[index]]$source_files <- publication_source_files(
      figures[[index]]$sources
    )
    figures[[index]]$metadata <- metadata[[index]]
  }
  figures
}

reporting_ensure_parent <- function(path) {
  parent <- dirname(path)
  if (!dir.exists(parent) && !dir.create(parent, recursive = TRUE)) {
    stop("cannot create reporting directory: ", parent)
  }
  invisible(parent)
}

reporting_atomic_csv <- function(value, path) {
  reporting_ensure_parent(path)
  partial <- paste0(path, ".partial")
  if (file.exists(partial)) unlink(partial)
  on.exit(if (file.exists(partial)) unlink(partial), add = TRUE)
  utils::write.csv(value, partial, row.names = FALSE, na = "")
  if (!file.rename(partial, path)) stop("atomic CSV rename failed")
  invisible(path)
}

reporting_atomic_json <- function(value, path) {
  reporting_ensure_parent(path)
  partial <- paste0(path, ".partial")
  if (file.exists(partial)) unlink(partial)
  on.exit(if (file.exists(partial)) unlink(partial), add = TRUE)
  jsonlite::write_json(
    value, partial, auto_unbox = TRUE, null = "null", digits = 17, pretty = TRUE
  )
  if (!file.rename(partial, path)) stop("atomic JSON rename failed")
  invisible(path)
}

atomic_write_parquet <- function(value, path) {
  reporting_ensure_parent(path)
  partial <- paste0(path, ".partial")
  if (file.exists(partial)) unlink(partial)
  on.exit(if (file.exists(partial)) unlink(partial), add = TRUE)
  arrow::write_parquet(value, partial)
  if (!file.rename(partial, path)) stop("atomic Parquet rename failed")
  invisible(path)
}

atomic_write_graphic <- function(
  plot, path, open_device, finalize_graphic = function(path) invisible(path)
) {
  reporting_ensure_parent(path)
  partial <- paste0(path, ".partial")
  if (file.exists(partial)) unlink(partial)
  on.exit(if (file.exists(partial)) unlink(partial), add = TRUE)
  device_open <- FALSE
  on.exit(if (device_open) grDevices::dev.off(), add = TRUE)
  open_device(partial)
  device_open <- TRUE
  print(plot)
  grDevices::dev.off()
  device_open <- FALSE
  finalize_graphic(partial)
  if (!file.rename(partial, path)) stop("atomic graphic rename failed")
  invisible(path)
}

png_insert_srgb_chunk <- function(path) {
  payload <- readBin(path, what = "raw", n = file.info(path)$size)
  signature <- as.raw(c(137L, 80L, 78L, 71L, 13L, 10L, 26L, 10L))
  if (length(payload) < 8L || !identical(payload[seq_len(8L)], signature)) {
    stop("cannot tag invalid PNG as sRGB")
  }
  offset <- 9L
  idat_offset <- NA_integer_
  while (offset <= length(payload)) {
    if (offset + 11L > length(payload)) stop("truncated PNG chunk")
    size_bytes <- as.integer(payload[offset + 0:3])
    chunk_size <- sum(size_bytes * 256^(3:0))
    chunk_type <- rawToChar(payload[offset + 4:7])
    chunk_end <- offset + 11L + chunk_size
    if (chunk_end > length(payload)) stop("truncated PNG chunk data")
    if (identical(chunk_type, "sRGB")) return(invisible(path))
    if (identical(chunk_type, "IDAT")) {
      idat_offset <- offset
      break
    }
    offset <- chunk_end + 1L
  }
  if (is.na(idat_offset)) stop("PNG has no IDAT chunk")
  chunk_type <- charToRaw("sRGB")
  chunk_data <- as.raw(0L)
  crc_hex <- digest::digest(
    c(chunk_type, chunk_data), algo = "crc32", serialize = FALSE
  )
  starts <- c(1L, 3L, 5L, 7L)
  crc <- as.raw(strtoi(substring(crc_hex, starts, starts + 1L), base = 16L))
  srgb_chunk <- c(
    as.raw(c(0L, 0L, 0L, 1L)), chunk_type, chunk_data, crc
  )
  tagged <- c(
    payload[seq_len(idat_offset - 1L)],
    srgb_chunk,
    payload[idat_offset:length(payload)]
  )
  writeBin(tagged, path)
  invisible(path)
}

atomic_write_pdf <- function(plot, path, width_mm, height_mm) {
  atomic_write_graphic(plot, path, function(partial) {
    grDevices::cairo_pdf(
      partial,
      width = width_mm / 25.4,
      height = height_mm / 25.4,
      family = "Arial",
      bg = "white",
      onefile = TRUE
    )
  })
}

atomic_write_svg <- function(plot, path, width_mm, height_mm) {
  atomic_write_graphic(plot, path, function(partial) {
    svglite::svglite(
      partial,
      width = width_mm / 25.4,
      height = height_mm / 25.4,
      bg = "white",
      system_fonts = list(sans = "Arial")
    )
  })
}

atomic_write_png <- function(plot, path, width_mm, height_mm) {
  atomic_write_graphic(
    plot,
    path,
    function(partial) {
      ragg::agg_png(
        partial,
        width = width_mm,
        height = height_mm,
        units = "mm",
        res = 600,
        background = "white",
        scaling = 1
      )
    },
    finalize_graphic = png_insert_srgb_chunk
  )
}

reporting_atomic_text <- function(lines, path) {
  reporting_ensure_parent(path)
  partial <- paste0(path, ".partial")
  if (file.exists(partial)) unlink(partial)
  on.exit(if (file.exists(partial)) unlink(partial), add = TRUE)
  writeLines(lines, partial, useBytes = TRUE)
  if (!file.rename(partial, path)) stop("atomic text rename failed")
  invisible(path)
}

publication_captions_markdown <- function(metadata) {
  lines <- c(
    "# Publication figure captions",
    "",
    "Standalone captions for the fixed seven-figure main-text set."
  )
  for (index in seq_along(metadata)) {
    lines <- c(
      lines,
      "",
      sprintf("## Figure %d", index),
      "",
      metadata[[index]]$caption
    )
  }
  lines
}

render_reporting_outputs <- function(
  inputs, output_root, fail_after_artifact = NULL
) {
  generator_commit <- reporting_git_commit()
  artifact_count <- 0L
  completed_artifact <- function() {
    artifact_count <<- artifact_count + 1L
    if (!is.null(fail_after_artifact) && artifact_count >= fail_after_artifact) {
      stop("injected reporting failure after artifact ", artifact_count)
    }
  }
  tables <- publication_tables(inputs)
  table_paths <- vapply(seq_along(tables), function(index) {
    path <- file.path(output_root, "tables", sprintf("table_%d.csv", index))
    reporting_atomic_csv(tables[[index]], path)
    completed_artifact()
    path
  }, character(1L))
  figures <- publication_figures(inputs)
  figure_paths <- lapply(seq_along(figures), function(index) {
    figure_id <- sprintf("figure_%d", index)
    pdf_path <- file.path(output_root, "figures", paste0(figure_id, ".pdf"))
    svg_path <- file.path(output_root, "figures", paste0(figure_id, ".svg"))
    png_path <- file.path(output_root, "figures", paste0(figure_id, ".png"))
    data_path <- file.path(output_root, "figures", "data", paste0(figure_id, ".parquet"))
    provenance_path <- file.path(
      output_root, "figures", paste0(figure_id, ".provenance.json")
    )
    canvas <- figures[[index]]$metadata$target_canvas_mm
    atomic_write_parquet(figures[[index]]$data, data_path)
    completed_artifact()
    atomic_write_svg(
      figures[[index]]$plot, svg_path, canvas$width, canvas$height
    )
    completed_artifact()
    atomic_write_pdf(
      figures[[index]]$plot, pdf_path, canvas$width, canvas$height
    )
    completed_artifact()
    atomic_write_png(
      figures[[index]]$plot, png_path, canvas$width, canvas$height
    )
    completed_artifact()
    sources <- figures[[index]]$sources
    source_hashes <- inputs$source_table_hashes[sources]
    rendered_paths <- c(svg = svg_path, pdf = pdf_path, png = png_path)
    reporting_atomic_json(
      list(
        figure_id = figure_id,
        upstream_model_git_commit = as.character(inputs$run_context$git_commit),
        reporting_git_commit = generator_commit,
        evidence_policy_sha256 = as.character(inputs$evidence_policy_sha256),
        source_table_hashes = source_hashes,
        figure_data_sha256 = digest::digest(
          file = data_path, algo = "sha256", serialize = FALSE
        ),
        rendered_files_sha256 = as.list(vapply(
          rendered_paths,
          digest::digest,
          character(1L),
          algo = "sha256", serialize = FALSE, file = TRUE
        )),
        plotted_source_data_hash = attr(
          figures[[index]]$plot, "source_data_hash"
        )
      ),
      provenance_path
    )
    completed_artifact()
    rendered_paths
  })
  metadata <- lapply(figures, `[[`, "metadata")
  captions_path <- file.path(output_root, "figures", "figure_captions.md")
  manifest_path <- file.path(output_root, "figures", "figure_manifest.json")
  reporting_atomic_text(publication_captions_markdown(metadata), captions_path)
  completed_artifact()
  reporting_atomic_json(
    list(
      schema_version = "1.0",
      main_figure_count = length(metadata),
      captions_markdown_sha256 = digest::digest(
        file = captions_path, algo = "sha256", serialize = FALSE
      ),
      figures = metadata
    ),
    manifest_path
  )
  completed_artifact()
  list(
    tables = table_paths,
    figures = figure_paths,
    manifest = manifest_path,
    captions = captions_path
  )
}

expected_publication_files <- function() {
  table_files <- file.path("tables", sprintf("table_%d.csv", seq_len(8L)))
  figure_ids <- sprintf("figure_%d", seq_len(7L))
  c(
    table_files,
    file.path("figures", paste0(figure_ids, ".pdf")),
    file.path("figures", paste0(figure_ids, ".svg")),
    file.path("figures", paste0(figure_ids, ".png")),
    file.path("figures", paste0(figure_ids, ".provenance.json")),
    file.path("figures", "data", paste0(figure_ids, ".parquet")),
    file.path("figures", "figure_manifest.json"),
    file.path("figures", "figure_captions.md")
  )
}

validate_staged_publication_set <- function(output_root) {
  observed <- sort(list.files(
    output_root, recursive = TRUE, all.files = TRUE, no.. = TRUE
  ))
  expected <- sort(expected_publication_files())
  if (!identical(observed, expected)) {
    stop(
      "unexpected publication files; missing=",
      paste(setdiff(expected, observed), collapse = ","),
      "; extra=", paste(setdiff(observed, expected), collapse = ",")
    )
  }
  paths <- file.path(output_root, expected)
  if (any(file.info(paths)$size <= 0L)) stop("publication contains empty files")
  parquet_paths <- paths[grepl("[.]parquet$", paths)]
  invisible(lapply(parquet_paths, arrow::read_parquet))
  json_paths <- paths[grepl("[.]json$", paths)]
  invisible(lapply(json_paths, jsonlite::read_json, simplifyVector = TRUE))
  invisible(TRUE)
}

install_staged_publication <- function(stage_root, output_root) {
  backup_root <- tempfile("reporting-backup-", tmpdir = output_root)
  if (!dir.create(backup_root)) stop("cannot create publication rollback directory")
  committed <- FALSE
  moved_new <- character()
  on.exit({
    if (!committed) {
      for (name in rev(moved_new)) {
        current <- file.path(output_root, name)
        if (dir.exists(current)) unlink(current, recursive = TRUE)
      }
      for (name in c("tables", "figures")) {
        backup <- file.path(backup_root, name)
        if (dir.exists(backup)) file.rename(backup, file.path(output_root, name))
      }
    }
    if (dir.exists(backup_root)) unlink(backup_root, recursive = TRUE)
  }, add = TRUE)
  for (name in c("tables", "figures")) {
    current <- file.path(output_root, name)
    if (dir.exists(current) && !file.rename(current, file.path(backup_root, name))) {
      stop("cannot stage prior publication directory: ", name)
    }
  }
  for (name in c("tables", "figures")) {
    staged <- file.path(stage_root, name)
    if (!file.rename(staged, file.path(output_root, name))) {
      stop("cannot install staged publication directory: ", name)
    }
    moved_new <- c(moved_new, name)
  }
  committed <- TRUE
  invisible(TRUE)
}

publish_reporting_transaction <- function(
  inputs, output_root, fail_after_artifact = NULL
) {
  if (!dir.exists(output_root) && !dir.create(output_root, recursive = TRUE)) {
    stop("cannot create reporting output root")
  }
  revoke_reporting_manifest(output_root)
  stage_root <- tempfile(".reporting-stage-", tmpdir = output_root)
  if (!dir.create(stage_root)) stop("cannot create same-filesystem reporting stage")
  on.exit(if (dir.exists(stage_root)) unlink(stage_root, recursive = TRUE), add = TRUE)
  result <- render_reporting_outputs(
    inputs, stage_root, fail_after_artifact = fail_after_artifact
  )
  validate_staged_publication_set(stage_root)
  install_staged_publication(stage_root, output_root)
  list(
    tables = file.path(output_root, "tables", basename(result$tables)),
    figures = lapply(result$figures, function(paths) {
      file.path(output_root, "figures", basename(paths))
    }),
    manifest = file.path(output_root, "figures", basename(result$manifest)),
    captions = file.path(output_root, "figures", basename(result$captions))
  )
}

revoke_reporting_manifest <- function(output_root) {
  manifest_path <- file.path(output_root, "run_manifest.json")
  if (file.exists(manifest_path) && unlink(manifest_path) != 0L) {
    stop("cannot revoke prior successful run manifest")
  }
  invisible(manifest_path)
}

run_reporting <- function(output_root) {
  revoke_reporting_manifest(output_root)
  inputs <- load_reporting_inputs(output_root)
  publish_reporting_transaction(inputs, output_root)
}
