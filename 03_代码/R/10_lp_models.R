main_controls <- c(
  "renewable_energy_consumption_share_a",
  "trade_openness_a",
  "industry_value_added_share_a",
  "gdp_per_capita_a"
)

MODEL_TERMS <- c("gimc_a", "gimc_gad_a")

model_columns <- c(
  "economy_id", "treatment_time", "delta_outcome",
  "gimc_a", "gimc_gad_a", "gad_a", "z_a", "z_gad_a",
  main_controls
)

require_cell_field <- function(cell, field) {
  value <- cell[[field]]
  if (is.null(value) || length(value) != 1L || is.na(value)) {
    stop("cell is missing scalar field: ", field)
  }
  value
}

prepare_model_frame <- function(d, cell) {
  if (!is.data.frame(d)) {
    stop("model input must be a data frame")
  }
  horizon <- as.integer(require_cell_field(cell, "horizon"))
  if (horizon == 0L) {
    stop("horizon zero is descriptive-only and cannot be estimated")
  }
  if (horizon < 0L) {
    stop("model horizon must be positive")
  }
  missing <- setdiff(model_columns, names(d))
  if (length(missing)) {
    stop("model frame is missing columns: ", paste(missing, collapse = ", "))
  }
  if (nrow(d) == 0L) {
    stop("model frame must not be empty")
  }

  identity_fields <- c(
    outcome_id = "outcome_id",
    horizon = "horizon",
    gad_version = "gad_version",
    sample_version = "sample_version",
    analysis_family = "analysis_family"
  )
  for (field in names(identity_fields)) {
    column <- identity_fields[[field]]
    if (column %in% names(d) && !all(d[[column]] == require_cell_field(cell, field))) {
      stop("model frame cell identity mismatch: ", field)
    }
  }

  if (anyNA(d[model_columns])) {
    stop("model frame contains missing required values")
  }
  if (any(!nzchar(as.character(d$economy_id)))) {
    stop("model frame contains an empty economy_id")
  }
  numeric_columns <- setdiff(model_columns, "economy_id")
  non_numeric <- numeric_columns[!vapply(d[numeric_columns], is.numeric, logical(1L))]
  if (length(non_numeric)) {
    stop("model frame has non-numeric columns: ", paste(non_numeric, collapse = ", "))
  }
  if (any(!is.finite(as.matrix(d[numeric_columns])))) {
    stop("model frame contains nonfinite required values")
  }
  if (any(d$treatment_time < 2000L | d$treatment_time > 2022L)) {
    stop("model frame treatment_time is outside the frozen 2000-2022 period")
  }
  key <- paste(as.character(d$economy_id), d$treatment_time, sep = "\r")
  if (anyDuplicated(key)) {
    stop("duplicate economy-time key in model frame")
  }

  result <- d[order(as.character(d$economy_id), d$treatment_time), , drop = FALSE]
  rownames(result) <- NULL
  result
}

normalize_fixest_names <- function(value) {
  names(value) <- sub("^fit_", "", names(value))
  value
}

make_model_bundle <- function(fit, estimator, cell, run_context, data = NULL) {
  validate_run_context(run_context)
  coefficients <- normalize_fixest_names(stats::coef(fit))
  wanted <- c("gimc_a", "gimc_gad_a")
  if (!all(wanted %in% names(coefficients))) {
    stop("fitted model is missing a registered endogenous coefficient")
  }
  covariance <- stats::vcov(fit)
  rownames(covariance) <- sub("^fit_", "", rownames(covariance))
  colnames(covariance) <- sub("^fit_", "", colnames(covariance))
  if (!all(wanted %in% rownames(covariance)) || !all(wanted %in% colnames(covariance))) {
    stop("fitted model covariance is missing a registered term")
  }
  covariance <- covariance[wanted, wanted, drop = FALSE]
  coefficients <- coefficients[wanted]
  standard_errors <- sqrt(diag(covariance))
  if (
    any(!is.finite(coefficients)) || any(!is.finite(covariance)) ||
      any(!is.finite(standard_errors)) || any(standard_errors <= 0)
  ) {
    stop("fitted model produced invalid estimates")
  }
  if (is.null(data)) {
    data <- stats::model.frame(fit)
  }
  n <- as.integer(stats::nobs(fit))
  economies <- if ("economy_id" %in% names(data)) {
    length(unique(as.character(data$economy_id)))
  } else {
    length(unique(as.character(fixest::obs(fit, sample = "original"))))
  }
  clusters <- as.integer(economies)
  reference_df <- cluster_reference_df(clusters)
  critical <- stats::qt(0.975, df = reference_df)
  list(
    fit = fit,
    estimator = estimator,
    cell = cell,
    run_context = run_context,
    coefficients = coefficients,
    covariance = covariance,
    std_error = standard_errors,
    conf_low = coefficients - critical * standard_errors,
    conf_high = coefficients + critical * standard_errors,
    p_value = mapply(
      cluster_t_p_value,
      coefficients,
      standard_errors,
      MoreArgs = list(reference_df = reference_df)
    ),
    fixed_effects = c("economy_id", "treatment_time"),
    cluster = "economy_id",
    n = n,
    economies = as.integer(economies),
    clusters = clusters,
    ssc_config = FROZEN_SSC_CONFIG,
    reference_distribution = "cluster_t",
    reference_df = reference_df,
    data = data
  )
}

fit_lp_fe <- function(d, cell, run_context) {
  prepared <- prepare_model_frame(d, cell)
  estimation <- prepared
  estimation$economy_id <- match(
    as.character(estimation$economy_id),
    sort(unique(as.character(estimation$economy_id)))
  )
  fml <- stats::as.formula(paste0(
    "delta_outcome ~ gimc_a + gimc_gad_a + gad_a + ",
    paste(main_controls, collapse = " + "),
    " | economy_id + treatment_time"
  ))
  fit <- fixest::feols(
    fml,
    data = estimation,
    vcov = ~economy_id,
    ssc = frozen_fixest_ssc(),
    fixef.rm = "none",
    notes = FALSE
  )
  make_model_bundle(fit, "lp_fe", cell, run_context, prepared)
}

fit_lp_iv <- function(d, cell, run_context) {
  prepared <- prepare_model_frame(d, cell)
  estimation <- prepared
  estimation$economy_id <- match(
    as.character(estimation$economy_id),
    sort(unique(as.character(estimation$economy_id)))
  )
  fml <- stats::as.formula(paste0(
    "delta_outcome ~ gad_a + ", paste(main_controls, collapse = " + "),
    " | economy_id + treatment_time",
    " | gimc_a + gimc_gad_a ~ z_a + z_gad_a"
  ))
  fit <- fixest::feols(
    fml,
    data = estimation,
    vcov = ~economy_id,
    ssc = frozen_fixest_ssc(),
    fixef.rm = "none",
    notes = FALSE
  )
  make_model_bundle(fit, "lp_iv", cell, run_context, prepared)
}

compute_marginal_effects <- function(bundle, gad_values, gad_quantiles = NULL) {
  if (!all(c("gimc_a", "gimc_gad_a") %in% names(bundle$coefficients))) {
    stop("model bundle is missing marginal-effect coefficients")
  }
  if (!is.numeric(gad_values) || !length(gad_values) || any(!is.finite(gad_values))) {
    stop("gad_values must contain finite numeric values")
  }
  if (is.null(gad_quantiles)) {
    gad_quantiles <- seq_along(gad_values) - 1L
  }
  if (length(gad_quantiles) != length(gad_values)) {
    stop("gad_quantiles and gad_values must have equal length")
  }
  beta <- unname(bundle$coefficients[["gimc_a"]])
  theta <- unname(bundle$coefficients[["gimc_gad_a"]])
  covariance <- bundle$covariance[c("gimc_a", "gimc_gad_a"), c("gimc_a", "gimc_gad_a")]
  variance <- covariance[1L, 1L] +
    gad_values^2 * covariance[2L, 2L] +
    2 * gad_values * covariance[1L, 2L]
  tolerance <- 1e-12 * max(1, max(abs(covariance)))
  if (any(variance < -tolerance)) {
    stop("marginal-effect covariance produced a negative variance")
  }
  variance <- pmax(variance, 0)
  standard_error <- sqrt(variance)
  estimate <- beta + gad_values * theta
  reference_df <- bundle$reference_df
  if (
    length(reference_df) != 1L || !is.finite(reference_df) ||
      reference_df <= 0
  ) {
    stop("marginal effects require the frozen positive cluster-t df")
  }
  p_value <- ifelse(
    standard_error > 0,
    2 * stats::pt(-abs(estimate / standard_error), df = reference_df),
    as.numeric(estimate == 0)
  )
  critical <- stats::qt(0.975, df = reference_df)
  data.frame(
    gad_quantile = as.numeric(gad_quantiles),
    gad_value = as.numeric(gad_values),
    estimate = as.numeric(estimate),
    std_error = as.numeric(standard_error),
    conf_low = as.numeric(estimate - critical * standard_error),
    conf_high = as.numeric(estimate + critical * standard_error),
    p_value = as.numeric(p_value)
  )
}
