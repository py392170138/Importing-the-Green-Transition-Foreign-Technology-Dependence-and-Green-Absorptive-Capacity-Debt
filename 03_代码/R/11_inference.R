classify_inference <- function(n_clusters, rank) {
  if (rank < 2L) {
    return("fail_rank_deficient")
  }
  if (n_clusters < 20L) {
    return("exploratory_lt20_clusters")
  }
  if (n_clusters < 30L) {
    return("wild_bootstrap_required")
  }
  "cluster_robust"
}

validate_julia_runtime <- function(
  project_root,
  expected_julia = "1.12.7",
  expected_wild_boot_tests = "0.9.8"
) {
  executable <- Sys.which("julia")
  if (!nzchar(executable)) {
    stop("Julia is required for LP-IV wild-bootstrap inference")
  }
  project <- file.path(project_root, "julia")
  manifest <- file.path(project, "Manifest.toml")
  if (!file.exists(file.path(project, "Project.toml")) || !file.exists(manifest)) {
    stop("the frozen Julia analysis environment is missing")
  }
  version_output <- system2(
    executable,
    "--version",
    stdout = TRUE,
    stderr = TRUE
  )
  expected_output <- paste("julia version", expected_julia)
  if (!identical(version_output, expected_output)) {
    stop("Julia runtime version does not match the frozen environment")
  }
  text <- paste(readLines(manifest, warn = FALSE), collapse = "\n")
  if (!grepl(
    paste0('julia_version = "', expected_julia, '"'),
    text,
    fixed = TRUE
  )) {
    stop("Julia manifest runtime version is not frozen")
  }
  package_block <- regmatches(
    text,
    regexpr(
      "\\[\\[deps\\.WildBootTests\\]\\][\\s\\S]*?(?=\\n\\[\\[deps\\.|$)",
      text,
      perl = TRUE
    )
  )
  if (
    !length(package_block) ||
      !grepl(
        paste0('version = "', expected_wild_boot_tests, '"'),
        package_block,
        fixed = TRUE
      )
  ) {
    stop("WildBootTests.jl version does not match the frozen manifest")
  }
  invisible(list(
    julia_version = expected_julia,
    wild_boot_tests_version = expected_wild_boot_tests,
    project = project
  ))
}

fixed_effect_design <- function(d) {
  n <- nrow(d)
  controls <- as.matrix(d[c("gad_a", main_controls)])
  economy <- as.character(d$economy_id)
  treatment_time <- d$treatment_time
  economy_levels <- sort(unique(economy))
  time_levels <- sort(unique(treatment_time))
  columns <- list(rep(1, n))
  columns <- c(columns, lapply(seq_len(ncol(controls)), function(index) {
    controls[, index]
  }))
  if (length(economy_levels) > 1L) {
    columns <- c(columns, lapply(economy_levels[-1L], function(level) {
      as.numeric(economy == level)
    }))
  }
  if (length(time_levels) > 1L) {
    columns <- c(columns, lapply(time_levels[-1L], function(level) {
      as.numeric(treatment_time == level)
    }))
  }
  do.call(cbind, columns)
}

qr_residualize <- function(values, design) {
  decomposition <- qr(design, LAPACK = TRUE)
  r <- qr.R(decomposition)
  diagonal <- abs(diag(r))
  if (!length(diagonal)) {
    return(values)
  }
  tolerance <- max(dim(design)) * .Machine$double.eps * max(diagonal)
  rank <- as.integer(sum(diagonal > tolerance))
  q <- qr.Q(decomposition, complete = FALSE)
  basis <- q[, seq_len(rank), drop = FALSE]
  result <- values - basis %*% crossprod(basis, values)
  result[abs(result) < 1e-14] <- 0
  result
}

cluster_covariance <- function(x, residual, clusters) {
  n <- nrow(x)
  k <- ncol(x)
  levels <- sort(unique(as.character(clusters)))
  g <- length(levels)
  if (g <= 1L || n <= k) {
    stop("cluster covariance requires multiple clusters and residual degrees of freedom")
  }
  bread <- solve(crossprod(x))
  meat <- matrix(0, nrow = k, ncol = k)
  cluster_values <- as.character(clusters)
  for (level in levels) {
    selected <- cluster_values == level
    score <- crossprod(x[selected, , drop = FALSE], residual[selected])
    meat <- meat + tcrossprod(as.numeric(score))
  }
  correction <- (g / (g - 1)) * ((n - 1) / (n - k))
  correction * bread %*% meat %*% bread
}

first_stage_for_outcome <- function(response, instruments, clusters) {
  beta <- as.numeric(solve(crossprod(instruments), crossprod(instruments, response)))
  residual <- as.numeric(response - instruments %*% beta)
  covariance <- cluster_covariance(instruments, residual, clusters)
  restricted_sse <- sum(response^2)
  if (!is.finite(restricted_sse) || restricted_sse <= .Machine$double.eps) {
    stop("first-stage endogenous residual has zero variation")
  }
  partial_r2 <- min(1, max(0, 1 - sum(residual^2) / restricted_sse))
  effective_f <- as.numeric(crossprod(beta, solve(covariance, beta))) /
    ncol(instruments)
  c(partial_r2 = partial_r2, effective_f = effective_f)
}

compute_first_stage_metrics <- function(d) {
  prepared <- prepare_model_frame(d, list(
    outcome_id = "synthetic",
    horizon = 1L,
    gad_version = "synthetic",
    sample_version = "synthetic",
    analysis_family = "synthetic"
  ))
  values <- as.matrix(prepared[c("gimc_a", "gimc_gad_a", "z_a", "z_gad_a")])
  residualized <- qr_residualize(values, fixed_effect_design(prepared))
  endogenous <- residualized[, 1:2, drop = FALSE]
  instruments <- residualized[, 3:4, drop = FALSE]
  instrument_rank <- matrix_rank(instruments)
  cross_moment <- crossprod(instruments, endogenous) / nrow(prepared)
  cross_moment_rank <- matrix_rank(cross_moment)
  rank <- min(instrument_rank, cross_moment_rank)
  singular_values <- svd(cross_moment, nu = 0L, nv = 0L)$d
  condition_number <- if (rank == 2L && min(singular_values) > 0) {
    max(singular_values) / min(singular_values)
  } else {
    Inf
  }
  first_gimc <- first_stage_for_outcome(
    endogenous[, 1L], instruments, prepared$economy_id
  )
  first_interaction <- first_stage_for_outcome(
    endogenous[, 2L], instruments, prepared$economy_id
  )
  list(
    instrument_rank = as.integer(instrument_rank),
    cross_moment_rank = as.integer(cross_moment_rank),
    rank = as.integer(rank),
    condition_number = as.numeric(condition_number),
    partial_r2_gimc = unname(first_gimc[["partial_r2"]]),
    partial_r2_interaction = unname(first_interaction[["partial_r2"]]),
    effective_f_gimc = unname(first_gimc[["effective_f"]]),
    effective_f_interaction = unname(first_interaction[["effective_f"]])
  )
}

run_wild_fe <- function(fit, parameter, seed = 20260820L, draws = 9999L) {
  set.seed(seed)
  dqrng::dqset.seed(seed)
  fwildclusterboot::boottest(
    fit,
    clustid = "economy_id",
    param = parameter,
    B = draws,
    conf_int = TRUE,
    fe = "economy_id"
  )
}

run_wild_iv <- function(d, parameter, seed = 20260820L, draws = 9999L) {
  set.seed(seed)
  dqrng::dqset.seed(seed)
  d <- d[order(as.character(d$economy_id), d$treatment_time), , drop = FALSE]
  d$economy_id <- match(
    as.character(d$economy_id),
    sort(unique(as.character(d$economy_id)))
  )
  fit <- ivreg::ivreg(
    delta_outcome ~ gimc_a + gimc_gad_a + gad_a +
      renewable_energy_consumption_share_a + trade_openness_a +
      industry_value_added_share_a + gdp_per_capita_a +
      factor(economy_id) + factor(treatment_time) |
      z_a + z_gad_a + gad_a +
      renewable_energy_consumption_share_a + trade_openness_a +
      industry_value_added_share_a + gdp_per_capita_a +
      factor(economy_id) + factor(treatment_time),
    data = d
  )
  fwildclusterboot::boottest(
    fit,
    clustid = ~economy_id,
    param = parameter,
    B = draws,
    conf_int = TRUE
  )
}

extract_wild_result <- function(result) {
  if (!inherits(result, "boottest")) {
    stop("wild-bootstrap result does not inherit from boottest")
  }
  interval <- as.numeric(result$conf_int)
  p_value <- as.numeric(result$p_val)
  draws <- as.integer(result$boot_iter)
  estimate <- as.numeric(result$point_estimate)
  statistic <- as.numeric(result$t_stat)
  standard_error <- abs(estimate / statistic)
  finite_interval <- length(interval) == 2L && all(is.finite(interval))
  unbounded_interval <- length(interval) == 2L && identical(
    sort(interval),
    c(-Inf, Inf)
  )
  if (
    (!finite_interval && !unbounded_interval) ||
      length(p_value) != 1L || !is.finite(p_value) ||
      p_value < 0 || p_value > 1 ||
      length(draws) != 1L || is.na(draws) || draws <= 0L
      || length(estimate) != 1L || !is.finite(estimate)
      || length(standard_error) != 1L || !is.finite(standard_error)
      || standard_error <= 0
  ) {
    stop("wild-bootstrap result is incomplete")
  }
  list(
    estimate = estimate,
    std_error = standard_error,
    conf_low = if (finite_interval) min(interval) else NA_real_,
    conf_high = if (finite_interval) max(interval) else NA_real_,
    p_value = p_value,
    draws = draws,
    interval_status = if (finite_interval) "available" else "unbounded"
  )
}

center_marginal_frame <- function(d, gad_value) {
  if (length(gad_value) != 1L || !is.finite(gad_value)) {
    stop("marginal GAD value must be one finite number")
  }
  centered <- d
  centered$gad_centered <- centered$gad_a - gad_value
  centered$gimc_gad_centered <- centered$gimc_a * centered$gad_centered
  centered$z_gad_centered <- centered$z_a * centered$gad_centered
  centered
}

run_wild_fe_marginal <- function(
  d, gad_value, seed = 20260820L, draws = 9999L
) {
  centered <- center_marginal_frame(d, gad_value)
  centered$economy_id <- match(
    as.character(centered$economy_id),
    sort(unique(as.character(centered$economy_id)))
  )
  fml <- stats::as.formula(paste0(
    "delta_outcome ~ gimc_a + gimc_gad_centered + gad_centered + ",
    paste(main_controls, collapse = " + "),
    " | economy_id + treatment_time"
  ))
  fit <- fixest::feols(
    fml,
    data = centered,
    vcov = ~economy_id,
    fixef.rm = "none",
    notes = FALSE
  )
  run_wild_fe(fit, "gimc_a", seed = seed, draws = draws)
}

run_wild_iv_marginal <- function(
  d, gad_value, seed = 20260820L, draws = 9999L
) {
  set.seed(seed)
  dqrng::dqset.seed(seed)
  centered <- center_marginal_frame(d, gad_value)
  centered <- centered[
    order(as.character(centered$economy_id), centered$treatment_time),
    ,
    drop = FALSE
  ]
  centered$economy_id <- match(
    as.character(centered$economy_id),
    sort(unique(as.character(centered$economy_id)))
  )
  fit <- ivreg::ivreg(
    delta_outcome ~ gimc_a + gimc_gad_centered + gad_centered +
      renewable_energy_consumption_share_a + trade_openness_a +
      industry_value_added_share_a + gdp_per_capita_a +
      factor(economy_id) + factor(treatment_time) |
      z_a + z_gad_centered + gad_centered +
      renewable_energy_consumption_share_a + trade_openness_a +
      industry_value_added_share_a + gdp_per_capita_a +
      factor(economy_id) + factor(treatment_time),
    data = centered
  )
  fwildclusterboot::boottest(
    fit,
    clustid = ~economy_id,
    param = "gimc_a",
    B = draws,
    conf_int = TRUE
  )
}
