threshold_required_columns <- c(
  "economy_id", "treatment_time", "delta_outcome", "gimc_a", "gad_a",
  "renewable_energy_consumption_share_a", "trade_openness_a",
  "industry_value_added_share_a", "gdp_per_capita_a"
)

validate_threshold_spec <- function(spec) {
  required <- c(
    "outcome_id", "horizon", "gad_version", "sample_version",
    "percentiles", "minimum_regime_share", "criterion", "quantile_type",
    "tie_break", "seed"
  )
  missing <- setdiff(required, names(spec))
  if (length(missing)) {
    stop("threshold spec is missing: ", paste(missing, collapse = ", "))
  }
  percentiles <- as.integer(unlist(spec$percentiles))
  if (!identical(percentiles, 20:80)) {
    stop("threshold percentiles must be the literal integer grid 20:80")
  }
  if (
    !identical(as.character(spec$criterion), "minimum_two_way_fe_ssr") ||
      as.integer(spec$quantile_type) != 7L ||
      !identical(
        as.character(spec$tie_break),
        "lowest_percentile_then_lowest_q"
      ) ||
      as.numeric(spec$minimum_regime_share) != 0.20
  ) {
    stop("threshold selection rules differ from the frozen specification")
  }
  invisible(spec)
}

prepare_threshold_frame <- function(d) {
  if (!is.data.frame(d) || !nrow(d)) {
    stop("threshold input must be a non-empty data frame")
  }
  missing <- setdiff(threshold_required_columns, names(d))
  if (length(missing)) {
    stop("threshold frame is missing: ", paste(missing, collapse = ", "))
  }
  selected <- d[threshold_required_columns]
  if (anyNA(selected)) {
    stop("threshold frame contains missing required values")
  }
  numeric_columns <- setdiff(names(selected), "economy_id")
  if (any(!is.finite(as.matrix(selected[numeric_columns])))) {
    stop("threshold frame contains nonfinite required values")
  }
  if (anyDuplicated(paste(selected$economy_id, selected$treatment_time))) {
    stop("threshold frame has duplicate economy-time keys")
  }
  selected <- selected[order(
    as.character(selected$economy_id), selected$treatment_time
  ), , drop = FALSE]
  rownames(selected) <- NULL
  selected
}

threshold_candidate_fit <- function(d, q) {
  estimation <- d
  estimation$gimc_low <- estimation$gimc_a * as.integer(estimation$gad_a <= q)
  estimation$gimc_high <- estimation$gimc_a * as.integer(estimation$gad_a > q)
  estimation$economy_id <- match(
    as.character(estimation$economy_id),
    sort(unique(as.character(estimation$economy_id)))
  )
  fit <- fixest::feols(
    delta_outcome ~ gimc_low + gimc_high + gad_a +
      renewable_energy_consumption_share_a + trade_openness_a +
      industry_value_added_share_a + gdp_per_capita_a |
      economy_id + treatment_time,
    data = estimation,
    fixef.rm = "none",
    notes = FALSE,
    warn = FALSE
  )
  residual <- stats::residuals(fit)
  c(ssr = sum(residual^2), n = stats::nobs(fit))
}

threshold_candidate_grid <- function(d, spec) {
  percentiles <- as.integer(unlist(spec$percentiles))
  q <- as.numeric(stats::quantile(
    d$gad_a,
    probs = percentiles / 100,
    type = as.integer(spec$quantile_type),
    names = FALSE
  ))
  grid <- data.frame(percentile = percentiles, q = q)
  grid <- grid[!duplicated(grid$q), , drop = FALSE]
  grid$low_share <- vapply(
    grid$q, function(value) mean(d$gad_a <= value), numeric(1L)
  )
  grid$high_share <- vapply(
    grid$q, function(value) mean(d$gad_a > value), numeric(1L)
  )
  grid$eligible <- grid$low_share >= as.numeric(spec$minimum_regime_share) &
    grid$high_share >= as.numeric(spec$minimum_regime_share)
  grid$ssr <- NA_real_
  grid$n <- NA_integer_
  grid$status <- ifelse(grid$eligible, "pending", "regime_share_below_minimum")
  grid
}

choose_threshold <- function(d, spec) {
  validate_threshold_spec(spec)
  prepared <- prepare_threshold_frame(d)
  candidates <- threshold_candidate_grid(prepared, spec)
  for (index in which(candidates$eligible)) {
    fitted <- tryCatch(
      threshold_candidate_fit(prepared, candidates$q[[index]]),
      error = function(error) NULL
    )
    if (
      !is.null(fitted) && length(fitted) == 2L &&
        all(is.finite(fitted)) && fitted[["n"]] > 0
    ) {
      candidates$ssr[[index]] <- as.numeric(fitted[["ssr"]])
      candidates$n[[index]] <- as.integer(fitted[["n"]])
      candidates$status[[index]] <- "estimated"
    } else {
      candidates$status[[index]] <- "fit_failed"
    }
  }
  valid <- which(candidates$status == "estimated" & is.finite(candidates$ssr))
  if (!length(valid)) {
    stop("threshold grid produced no valid candidate")
  }
  ordering <- valid[order(
    candidates$ssr[valid],
    candidates$percentile[valid],
    candidates$q[valid]
  )]
  winner <- candidates[ordering[[1L]], , drop = FALSE]
  list(
    q = as.numeric(winner$q[[1L]]),
    percentile = as.integer(winner$percentile[[1L]]),
    low_share = as.numeric(winner$low_share[[1L]]),
    high_share = as.numeric(winner$high_share[[1L]]),
    criterion = as.character(spec$criterion),
    candidates = candidates
  )
}

resample_economy_blocks <- function(d, sampled_economies) {
  if (!length(sampled_economies)) {
    stop("economy bootstrap must sample at least one block")
  }
  blocks <- vector("list", length(sampled_economies))
  for (position in seq_along(sampled_economies)) {
    block <- d[as.character(d$economy_id) == sampled_economies[[position]], ,
      drop = FALSE
    ]
    if (!nrow(block)) {
      stop("economy bootstrap selected an unknown block")
    }
    block$economy_id <- sprintf("bootstrap_%04d", position)
    blocks[[position]] <- block
  }
  result <- do.call(rbind, blocks)
  rownames(result) <- NULL
  result
}

bootstrap_threshold <- function(d, spec, draws) {
  draws <- as.integer(draws)
  if (length(draws) != 1L || is.na(draws) || draws <= 0L) {
    stop("threshold bootstrap draws must be positive")
  }
  economies <- sort(unique(as.character(d$economy_id)))
  selected_percentiles <- rep(NA_real_, draws)
  selected_q <- rep(NA_real_, draws)
  set.seed(as.integer(spec$seed))
  for (draw in seq_len(draws)) {
    sampled <- sample(economies, length(economies), replace = TRUE)
    boot <- resample_economy_blocks(d, sampled)
    chosen <- tryCatch(
      choose_threshold(boot, spec),
      error = function(error) NULL
    )
    if (!is.null(chosen)) {
      selected_percentiles[[draw]] <- chosen$percentile
      selected_q[[draw]] <- chosen$q
    }
  }
  list(percentile = selected_percentiles, q = selected_q)
}

canonical_threshold_number <- function(value) {
  if (length(value) != 1L || !is.finite(value)) {
    stop("canonical threshold number must be one finite value")
  }
  text <- sprintf("%.17f", as.numeric(value))
  text <- sub("0+$", "", text)
  text <- sub("\\.$", "", text)
  if (text %in% c("-0", "")) "0" else text
}

canonical_threshold_value <- function(value) {
  if (is.data.frame(value)) {
    return(lapply(seq_len(nrow(value)), function(index) {
      canonical_threshold_value(as.list(value[index, , drop = FALSE]))
    }))
  }
  if (is.list(value)) {
    if (!is.null(names(value))) {
      keys <- sort(names(value))
      result <- setNames(vector("list", length(keys)), keys)
      for (key in keys) {
        result[key] <- list(canonical_threshold_value(value[[key]]))
      }
      return(result)
    }
    return(lapply(value, canonical_threshold_value))
  }
  if (is.numeric(value)) {
    return(lapply(value, function(item) {
      if (is.na(item)) NULL else canonical_threshold_number(item)
    }) |> (function(items) if (length(items) == 1L) items[[1L]] else items)())
  }
  if (is.logical(value) && length(value) == 1L && is.na(value)) {
    return(NULL)
  }
  value
}

canonical_threshold_registry_hash <- function(registry) {
  payload <- registry[setdiff(
    names(registry), c("registry_hash", "created_at_utc")
  )]
  normalized <- canonical_threshold_value(payload)
  canonical_list_sha256(normalized)
}

verify_frozen_threshold_registry <- function(
  registry, d, spec, run_context, bootstrap_draws
) {
  expected <- list(
    registry_id = "threshold_registry_v1",
    selection_outcome = as.character(spec$outcome_id),
    horizon = as.integer(spec$horizon),
    gad_version = as.character(spec$gad_version),
    sample_version = as.character(spec$sample_version),
    criterion = as.character(spec$criterion),
    quantile_type = as.integer(spec$quantile_type),
    tie_break = as.character(spec$tie_break),
    seed = as.integer(spec$seed),
    sample_hash = canonical_frame_sha256(prepare_threshold_frame(d)),
    input_authority_hash = as.character(run_context$input_authority_hash),
    bootstrap_draws = as.integer(bootstrap_draws)
  )
  mismatch <- names(expected)[!vapply(names(expected), function(field) {
    !is.null(registry[[field]]) && identical(
      as.character(registry[[field]]), as.character(expected[[field]])
    )
  }, logical(1L))]
  if (length(mismatch)) {
    stop(
      "frozen threshold registry mismatch: ", paste(mismatch, collapse = ", ")
    )
  }
  observed_hash <- canonical_threshold_registry_hash(registry)
  if (
    is.null(registry$registry_hash) ||
      !identical(as.character(registry$registry_hash), observed_hash)
  ) {
    stop("frozen threshold registry mismatch: registry_hash")
  }
  invisible(registry)
}

select_and_freeze_threshold <- function(
  d, path, spec, run_context, bootstrap_draws = 999L,
  minimum_valid_draws = 900L
) {
  validate_threshold_spec(spec)
  validate_run_context(run_context)
  prepared <- prepare_threshold_frame(d)
  bootstrap_draws <- as.integer(bootstrap_draws)
  minimum_valid_draws <- as.integer(minimum_valid_draws)
  if (file.exists(path)) {
    existing <- jsonlite::read_json(path, simplifyVector = TRUE)
    verify_frozen_threshold_registry(
      existing, prepared, spec, run_context, bootstrap_draws
    )
    return(existing)
  }
  selected <- choose_threshold(prepared, spec)
  boot <- bootstrap_threshold(prepared, spec, bootstrap_draws)
  valid <- is.finite(boot$percentile) & is.finite(boot$q)
  valid_percentiles <- boot$percentile[valid]
  valid_q <- boot$q[valid]
  bootstrap_ok <- length(valid_q) >= minimum_valid_draws
  bounds <- if (bootstrap_ok) {
    list(
      percentile_conf_low = unname(stats::quantile(
        valid_percentiles, 0.025, type = 7
      )),
      percentile_conf_high = unname(stats::quantile(
        valid_percentiles, 0.975, type = 7
      )),
      q_conf_low = unname(stats::quantile(valid_q, 0.025, type = 7)),
      q_conf_high = unname(stats::quantile(valid_q, 0.975, type = 7))
    )
  } else {
    list(
      percentile_conf_low = NULL, percentile_conf_high = NULL,
      q_conf_low = NULL, q_conf_high = NULL
    )
  }
  registry <- c(list(
    registry_id = "threshold_registry_v1",
    selection_outcome = as.character(spec$outcome_id),
    horizon = as.integer(spec$horizon),
    gad_version = as.character(spec$gad_version),
    sample_version = as.character(spec$sample_version),
    criterion = as.character(spec$criterion),
    quantile_type = as.integer(spec$quantile_type),
    tie_break = as.character(spec$tie_break),
    seed = as.integer(spec$seed),
    sample_hash = canonical_frame_sha256(prepared),
    input_authority_hash = as.character(run_context$input_authority_hash),
    q = selected$q,
    percentile = selected$percentile,
    low_share = selected$low_share,
    high_share = selected$high_share,
    bootstrap_draws = bootstrap_draws,
    bootstrap_valid_draws = as.integer(length(valid_q)),
    bootstrap_failed_draws = as.integer(bootstrap_draws - length(valid_q)),
    bootstrap_status = if (bootstrap_ok) {
      "available"
    } else {
      "insufficient_valid_draws"
    }
  ), bounds, list(
    candidates = selected$candidates,
    created_at_utc = as.character(run_context$created_at_utc)
  ))
  registry$registry_hash <- canonical_threshold_registry_hash(registry)
  atomic_write_json(registry, path)
  written <- jsonlite::read_json(path, simplifyVector = TRUE)
  verify_frozen_threshold_registry(
    written, prepared, spec, run_context, bootstrap_draws
  )
  written
}

threshold_first_stage_metrics <- function(d, q) {
  indicators <- cbind(
    low = as.integer(d$gad_a <= q),
    high = as.integer(d$gad_a > q)
  )
  endogenous <- cbind(
    gimc_low = d$gimc_a * indicators[, "low"],
    gimc_high = d$gimc_a * indicators[, "high"]
  )
  instruments <- cbind(
    z_low = d$z_a * indicators[, "low"],
    z_high = d$z_a * indicators[, "high"]
  )
  residualized <- qr_residualize(
    cbind(endogenous, instruments), fixed_effect_design(d)
  )
  endogenous_r <- residualized[, 1:2, drop = FALSE]
  instruments_r <- residualized[, 3:4, drop = FALSE]
  instrument_rank <- matrix_rank(instruments_r)
  cross_moment_rank <- matrix_rank(crossprod(instruments_r, endogenous_r))
  rank <- min(instrument_rank, cross_moment_rank)
  effective_f <- c(low = NA_real_, high = NA_real_)
  if (instrument_rank == 2L) {
    effective_f[["low"]] <- first_stage_for_outcome(
      endogenous_r[, 1L], instruments_r, d$economy_id
    )[["effective_f"]]
    effective_f[["high"]] <- first_stage_for_outcome(
      endogenous_r[, 2L], instruments_r, d$economy_id
    )[["effective_f"]]
  }
  status <- if (rank < 2L) {
    "fail_rank_deficient"
  } else if (min(effective_f) < 10) {
    "weak_reference_below_10"
  } else {
    "adequate_reference_10"
  }
  list(
    instrument_rank = as.integer(instrument_rank),
    cross_moment_rank = as.integer(cross_moment_rank),
    rank = as.integer(rank),
    effective_f_low = as.numeric(effective_f[["low"]]),
    effective_f_high = as.numeric(effective_f[["high"]]),
    status = status
  )
}

fit_threshold_iv <- function(d, cell, run_context, q) {
  if (length(q) != 1L || !is.finite(q)) {
    stop("threshold IV requires one finite frozen q")
  }
  prepared <- prepare_model_frame(d, cell)
  prepared$gimc_low <- prepared$gimc_a * as.integer(prepared$gad_a <= q)
  prepared$gimc_high <- prepared$gimc_a * as.integer(prepared$gad_a > q)
  prepared$z_low <- prepared$z_a * as.integer(prepared$gad_a <= q)
  prepared$z_high <- prepared$z_a * as.integer(prepared$gad_a > q)
  first_stage <- threshold_first_stage_metrics(prepared, q)
  if (first_stage$rank < 2L) {
    stop("threshold IV excluded-instrument system is rank deficient")
  }
  estimation <- prepared
  estimation$economy_id <- match(
    as.character(estimation$economy_id),
    sort(unique(as.character(estimation$economy_id)))
  )
  formula <- stats::as.formula(paste0(
    "delta_outcome ~ gad_a + ", paste(main_controls, collapse = " + "),
    " | economy_id + treatment_time",
    " | gimc_low + gimc_high ~ z_low + z_high"
  ))
  fit <- fixest::feols(
    formula,
    data = estimation,
    vcov = ~economy_id,
    ssc = frozen_fixest_ssc(),
    fixef.rm = "none",
    notes = FALSE
  )
  coefficients <- stats::coef(fit)
  names(coefficients) <- sub("^fit_", "", names(coefficients))
  covariance <- stats::vcov(fit)
  rownames(covariance) <- sub("^fit_", "", rownames(covariance))
  colnames(covariance) <- sub("^fit_", "", colnames(covariance))
  terms <- c("gimc_low", "gimc_high")
  if (!all(terms %in% names(coefficients)) ||
    !all(terms %in% rownames(covariance))) {
    stop("threshold IV fit omitted a frozen regime coefficient")
  }
  coefficients <- coefficients[terms]
  covariance <- covariance[terms, terms, drop = FALSE]
  standard_error <- sqrt(diag(covariance))
  if (any(!is.finite(coefficients)) || any(!is.finite(standard_error)) ||
    any(standard_error <= 0)) {
    stop("threshold IV fit produced invalid estimates")
  }
  list(
    fit = fit,
    estimator = "threshold_iv",
    cell = cell,
    run_context = run_context,
    q = as.numeric(q),
    coefficients = coefficients,
    covariance = covariance,
    std_error = standard_error,
    n = as.integer(nrow(prepared)),
    economies = as.integer(length(unique(prepared$economy_id))),
    clusters = as.integer(length(unique(prepared$economy_id))),
    first_stage = first_stage,
    ssc_config = FROZEN_SSC_CONFIG,
    reference_distribution = "cluster_t",
    reference_df = cluster_reference_df(length(unique(prepared$economy_id))),
    data = prepared
  )
}

threshold_inference_plan <- function(bundle, seed = 20260820L, draws = 9999L) {
  status <- if (bundle$clusters < 20L) {
    "exploratory_lt20_clusters"
  } else if (bundle$clusters < 30L) {
    "wild_bootstrap_required"
  } else {
    "cluster_robust"
  }
  list(
    status = status,
    parameters = c("gimc_low", "gimc_high", "high_minus_low"),
    seed = as.integer(seed),
    draws = as.integer(draws)
  )
}

prepare_threshold_wild_frame <- function(d, cell, q) {
  prepared <- prepare_model_frame(d, cell)
  prepared$gimc_low <- prepared$gimc_a * as.integer(prepared$gad_a <= q)
  prepared$gimc_high <- prepared$gimc_a * as.integer(prepared$gad_a > q)
  prepared$z_low <- prepared$z_a * as.integer(prepared$gad_a <= q)
  prepared$z_high <- prepared$z_a * as.integer(prepared$gad_a > q)
  prepared$economy_id <- match(
    as.character(prepared$economy_id),
    sort(unique(as.character(prepared$economy_id)))
  )
  prepared
}

fit_threshold_wild_iv <- function(d, cell, q) {
  prepared <- prepare_threshold_wild_frame(d, cell, q)
  fit <- ivreg::ivreg(
    delta_outcome ~ gimc_low + gimc_high + gad_a +
      renewable_energy_consumption_share_a + trade_openness_a +
      industry_value_added_share_a + gdp_per_capita_a +
      factor(economy_id) + factor(treatment_time) |
      z_low + z_high + gad_a +
      renewable_energy_consumption_share_a + trade_openness_a +
      industry_value_added_share_a + gdp_per_capita_a +
      factor(economy_id) + factor(treatment_time),
    data = prepared
  )
  list(fit = fit, data = prepared)
}

fit_threshold_difference_iv <- function(d, cell, q) {
  prepared <- prepare_threshold_wild_frame(d, cell, q)
  fit <- ivreg::ivreg(
    delta_outcome ~ gimc_a + gimc_high + gad_a +
      renewable_energy_consumption_share_a + trade_openness_a +
      industry_value_added_share_a + gdp_per_capita_a +
      factor(economy_id) + factor(treatment_time) |
      z_a + z_high + gad_a +
      renewable_energy_consumption_share_a + trade_openness_a +
      industry_value_added_share_a + gdp_per_capita_a +
      factor(economy_id) + factor(treatment_time),
    data = prepared
  )
  list(fit = fit, data = prepared)
}

blank_threshold_wild <- function() {
  list(
    estimate = NA_real_, std_error = NA_real_, conf_low = NA_real_,
    conf_high = NA_real_, p_value = NA_real_, draws = NA_integer_,
    seed = NA_real_, interval_status = "not_required"
  )
}

threshold_wild_results <- function(
  bundle, d, seed = 20260820L, draws = 9999L
) {
  plan <- threshold_inference_plan(bundle, seed = seed, draws = draws)
  result <- list(
    low = blank_threshold_wild(),
    high = blank_threshold_wild(),
    difference = blank_threshold_wild()
  )
  if (!identical(plan$status, "wild_bootstrap_required")) {
    return(result)
  }
  conventional_difference <- as.numeric(
    bundle$coefficients[["gimc_high"]] - bundle$coefficients[["gimc_low"]]
  )
  levels_fit <- fit_threshold_wild_iv(d, bundle$cell, bundle$q)
  difference_fit <- fit_threshold_difference_iv(d, bundle$cell, bundle$q)
  level_coefficients <- stats::coef(levels_fit$fit)
  difference_coefficient <- stats::coef(difference_fit$fit)[["gimc_high"]]
  if (
    stats::nobs(levels_fit$fit) != bundle$n ||
      stats::nobs(difference_fit$fit) != bundle$n ||
      max(abs(c(
        level_coefficients[["gimc_low"]] - bundle$coefficients[["gimc_low"]],
        level_coefficients[["gimc_high"]] - bundle$coefficients[["gimc_high"]],
        difference_coefficient - conventional_difference
      ))) > 1e-8
  ) {
    stop("threshold wild-bootstrap reparameterization changed sample or estimate")
  }
  raw <- list(
    low = levels_fit$fit,
    high = levels_fit$fit,
    difference = difference_fit$fit
  )
  parameters <- c(low = "gimc_low", high = "gimc_high", difference = "gimc_high")
  for (name in names(raw)) {
    set.seed(as.integer(seed))
    dqrng::dqset.seed(as.integer(seed))
    boot <- fwildclusterboot::boottest(
      raw[[name]],
      clustid = ~economy_id,
      param = parameters[[name]],
      B = as.integer(draws),
      conf_int = TRUE
    )
    result[[name]] <- c(
      extract_wild_result(boot),
      list(seed = as.numeric(seed))
    )
  }
  result
}

threshold_estimate_rows <- function(
  bundle, d, registry, wild_results = NULL,
  seed = 20260820L, draws = 9999L
) {
  required_registry <- c(
    "selection_outcome", "registry_hash", "sample_hash", "q"
  )
  if (length(setdiff(required_registry, names(registry)))) {
    stop("threshold estimate registry binding is incomplete")
  }
  if (!isTRUE(all.equal(as.numeric(bundle$q), as.numeric(registry$q)))) {
    stop("threshold fit q differs from frozen registry")
  }
  prepared <- prepare_model_frame(d, bundle$cell)
  regime_index <- list(
    low = prepared$gad_a <= bundle$q,
    high = prepared$gad_a > bundle$q
  )
  term_by_regime <- c(low = "gimc_low", high = "gimc_high")
  plan <- threshold_inference_plan(bundle, seed = seed, draws = draws)
  inference_status <- plan$status
  if (is.null(wild_results)) {
    wild_results <- threshold_wild_results(
      bundle, d, seed = seed, draws = draws
    )
  }
  reference_df <- cluster_reference_df(bundle$clusters)
  critical <- stats::qt(0.975, df = reference_df)
  covariance <- as.numeric(bundle$covariance["gimc_high", "gimc_low"])
  difference_estimate <- as.numeric(
    bundle$coefficients[["gimc_high"]] - bundle$coefficients[["gimc_low"]]
  )
  difference_variance <- as.numeric(
    bundle$covariance["gimc_high", "gimc_high"] +
      bundle$covariance["gimc_low", "gimc_low"] -
      2 * bundle$covariance["gimc_high", "gimc_low"]
  )
  if (!is.finite(difference_variance) || difference_variance <= 0) {
    stop("threshold high-minus-low contrast variance is invalid")
  }
  difference_std_error <- sqrt(difference_variance)
  difference_wild <- wild_results$difference
  metadata <- frozen_result_metadata(
    bundle$run_context,
    prepared,
    bundle$clusters,
    reference_distribution = "cluster_t",
    reference_df = reference_df,
    ssc_config = bundle$ssc_config
  )
  rows <- lapply(names(regime_index), function(regime) {
    term <- term_by_regime[[regime]]
    estimate <- as.numeric(bundle$coefficients[[term]])
    std_error <- as.numeric(bundle$std_error[[term]])
    values <- c(
      list(
        run_id = bundle$run_context$run_id,
        spec_id = bundle$run_context$spec_id,
        input_authority_hash = bundle$run_context$input_authority_hash,
        git_commit = bundle$run_context$git_commit,
        renv_lock_sha256 = bundle$run_context$renv_lock_sha256,
        evidence_policy_sha256 = bundle$run_context$evidence_policy_sha256,
        created_at_utc = bundle$run_context$created_at_utc,
        year_min = metadata$year_min,
        year_max = metadata$year_max,
        r_version = metadata$r_version,
        python_version = metadata$python_version,
        julia_version = metadata$julia_version,
        package_versions_json = metadata$package_versions_json,
        random_seed = metadata$random_seed,
        ssc_config = metadata$ssc_config,
        reference_distribution = metadata$reference_distribution,
        reference_df = metadata$reference_df,
        estimator = bundle$estimator,
        analysis_family = as.character(bundle$cell$analysis_family),
        outcome_id = as.character(bundle$cell$outcome_id),
        horizon = as.integer(bundle$cell$horizon),
        gad_version = as.character(bundle$cell$gad_version),
        sample_version = as.character(bundle$cell$sample_version)
      ),
      list(
        selection_outcome = as.character(registry$selection_outcome),
        registry_hash = as.character(registry$registry_hash),
        registry_sample_hash = as.character(registry$sample_hash),
        q = as.numeric(registry$q),
        regime = regime,
        estimate = estimate,
        std_error = std_error,
        conf_low = estimate - critical * std_error,
        conf_high = estimate + critical * std_error,
        p_value = cluster_t_p_value(estimate, std_error, reference_df),
        wild_estimate = wild_results[[regime]]$estimate,
        wild_std_error = wild_results[[regime]]$std_error,
        wild_conf_low = wild_results[[regime]]$conf_low,
        wild_conf_high = wild_results[[regime]]$conf_high,
        wild_p_value = wild_results[[regime]]$p_value,
        wild_inference_status = if (
          identical(wild_results[[regime]]$interval_status, "available")
        ) "wild_bootstrap_bounded" else if (
          identical(wild_results[[regime]]$interval_status, "unbounded")
        ) "wild_bootstrap_unbounded" else "not_required",
        wild_draws = wild_results[[regime]]$draws,
        wild_seed = wild_results[[regime]]$seed,
        low_high_covariance = covariance,
        difference_estimate = difference_estimate,
        difference_std_error = difference_std_error,
        difference_conf_low = difference_estimate - critical * difference_std_error,
        difference_conf_high = difference_estimate + critical * difference_std_error,
        difference_p_value = cluster_t_p_value(
          difference_estimate, difference_std_error, reference_df
        ),
        difference_wild_estimate = difference_wild$estimate,
        difference_wild_std_error = difference_wild$std_error,
        difference_wild_conf_low = difference_wild$conf_low,
        difference_wild_conf_high = difference_wild$conf_high,
        difference_wild_p_value = difference_wild$p_value,
        difference_wild_inference_status = if (
          identical(difference_wild$interval_status, "available")
        ) "wild_bootstrap_bounded" else if (
          identical(difference_wild$interval_status, "unbounded")
        ) "wild_bootstrap_unbounded" else "not_required",
        difference_wild_draws = difference_wild$draws,
        difference_wild_seed = difference_wild$seed,
        regime_n = as.integer(sum(regime_index[[regime]])),
        regime_share = as.numeric(mean(regime_index[[regime]])),
        n = bundle$n,
        economies = bundle$economies,
        clusters = bundle$clusters,
        first_stage_status = bundle$first_stage$status,
        inference_status = inference_status
      )
    )
    as.data.frame(values, stringsAsFactors = FALSE, check.names = FALSE)
  })
  result <- do.call(rbind, rows)
  rownames(result) <- NULL
  result
}
