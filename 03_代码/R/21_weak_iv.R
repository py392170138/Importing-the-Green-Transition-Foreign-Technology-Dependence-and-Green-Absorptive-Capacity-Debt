validate_ar_spec <- function(spec) {
  required <- c(
    "ar_grid_points_per_axis", "ar_initial_se_span",
    "ar_max_expansions", "alpha"
  )
  missing <- setdiff(required, names(spec))
  if (length(missing)) {
    stop("AR inference spec is missing: ", paste(missing, collapse = ", "))
  }
  if (
    as.integer(spec$ar_grid_points_per_axis) != 121L ||
      as.numeric(spec$ar_initial_se_span) != 6 ||
      as.integer(spec$ar_max_expansions) != 4L ||
      as.numeric(spec$alpha) != 0.05
  ) {
    stop("AR inference rules differ from the frozen specification")
  }
  invisible(spec)
}

ar_quadratic_coefficients <- function(a, b) {
  c(
    a^2,
    -2 * a * b[[1L]],
    -2 * a * b[[2L]],
    b[[1L]]^2,
    2 * b[[1L]] * b[[2L]],
    b[[2L]]^2
  )
}

ar_cross_coefficients <- function(a_left, b_left, a_right, b_right) {
  c(
    a_left * a_right,
    -(a_left * b_right[[1L]] + a_right * b_left[[1L]]),
    -(a_left * b_right[[2L]] + a_right * b_left[[2L]]),
    b_left[[1L]] * b_right[[1L]],
    b_left[[1L]] * b_right[[2L]] + b_left[[2L]] * b_right[[1L]],
    b_left[[2L]] * b_right[[2L]]
  )
}

prepare_ar_evaluator <- function(d) {
  prepared <- prepare_model_frame(d, list(
    outcome_id = "ar_internal", horizon = 1L,
    gad_version = "ar_internal", sample_version = "ar_internal",
    analysis_family = "ar_internal"
  ))
  y <- as.numeric(prepared$delta_outcome)
  x <- as.matrix(prepared[c("gimc_a", "gimc_gad_a")])
  z <- as.matrix(prepared[c("z_a", "z_gad_a")])
  residualized_z <- qr_residualize(
    z, fixed_effect_design(prepared)
  )
  if (matrix_rank(residualized_z) < 2L) {
    stop("AR region requires two linearly independent instruments")
  }

  auxiliary_formula <- stats::as.formula(paste0(
    "delta_outcome ~ z_a + z_gad_a + gad_a + ",
    paste(main_controls, collapse = " + "),
    " + factor(economy_id) + factor(treatment_time)"
  ))
  template_fit <- stats::lm(auxiliary_formula, data = prepared)
  template_vcov <- clubSandwich::vcovCR(
    template_fit,
    cluster = prepared$economy_id,
    type = "CR2"
  )
  aliased <- is.na(stats::coef(template_fit))
  design <- stats::model.matrix(template_fit)[, !aliased, drop = FALSE]
  coefficient_names <- names(stats::coef(template_fit))[!aliased]
  if (!identical(coefficient_names, colnames(template_vcov))) {
    stop("AR full-design coefficient order differs from the CR2 covariance")
  }
  target_indices <- match(c("z_a", "z_gad_a"), coefficient_names)
  if (anyNA(target_indices)) {
    stop("AR instruments are aliased in the full auxiliary design")
  }
  constraints <- matrix(
    0, nrow = 2L, ncol = length(coefficient_names)
  )
  constraints[cbind(seq_len(2L), target_indices)] <- 1
  template_test <- clubSandwich::Wald_test(
    template_fit,
    constraints = constraints,
    vcov = template_vcov,
    test = "HTZ"
  )
  df_denom <- as.numeric(template_test$df_denom[[1L]])
  delta <- as.numeric(template_test$delta[[1L]])
  if (!is.finite(df_denom) || df_denom <= 0 || !is.finite(delta) || delta <= 0) {
    stop("AR CR2/HTZ reference distribution is unavailable")
  }

  cluster_factor <- attr(template_vcov, "cluster")
  cluster_rows <- split(seq_len(nrow(prepared)), cluster_factor)
  estimation_matrices <- attr(template_vcov, "est_mats")
  adjustments <- attr(template_vcov, "adjustments")
  inverse_crossprod <- attr(template_vcov, "bread") /
    attr(template_vcov, "v_scale")
  target_bread <- inverse_crossprod[target_indices, , drop = FALSE]
  adjusted_matrices <- Map(
    function(estimation, adjustment) {
      target_bread %*% estimation %*% adjustment
    },
    estimation_matrices,
    adjustments
  )
  projection_y <- as.numeric(inverse_crossprod %*% crossprod(design, y))
  projection_x <- inverse_crossprod %*% crossprod(design, x)
  residual_y <- as.numeric(y - design %*% projection_y)
  residual_x <- x - design %*% projection_x

  score_intercepts <- vector("list", length(cluster_rows))
  score_slopes <- vector("list", length(cluster_rows))
  for (index in seq_along(cluster_rows)) {
    rows <- cluster_rows[[index]]
    adjusted <- adjusted_matrices[[index]]
    score_intercepts[[index]] <- as.numeric(adjusted %*% residual_y[rows])
    score_slopes[[index]] <- adjusted %*% residual_x[rows, , drop = FALSE]
  }
  m11 <- m12 <- m22 <- numeric(6L)
  for (index in seq_along(score_intercepts)) {
    intercept <- score_intercepts[[index]]
    slope <- score_slopes[[index]]
    m11 <- m11 + ar_quadratic_coefficients(intercept[[1L]], slope[1L, ])
    m22 <- m22 + ar_quadratic_coefficients(intercept[[2L]], slope[2L, ])
    m12 <- m12 + ar_cross_coefficients(
      intercept[[1L]], slope[1L, ], intercept[[2L]], slope[2L, ]
    )
  }
  list(
    coefficient_intercept = projection_y[target_indices],
    coefficient_slope = projection_x[target_indices, , drop = FALSE],
    meat_11 = m11,
    meat_12 = m12,
    meat_22 = m22,
    delta = delta,
    df_denom = df_denom,
    clusters = length(cluster_rows),
    observations = nrow(prepared)
  )
}

evaluate_ar_grid <- function(evaluator, beta_grid, theta_grid) {
  grid <- expand.grid(
    beta = as.numeric(beta_grid),
    theta = as.numeric(theta_grid),
    KEEP.OUT.ATTRS = FALSE,
    stringsAsFactors = FALSE
  )
  basis <- cbind(
    1,
    grid$beta,
    grid$theta,
    grid$beta^2,
    grid$beta * grid$theta,
    grid$theta^2
  )
  meat_11 <- as.numeric(basis %*% evaluator$meat_11)
  meat_12 <- as.numeric(basis %*% evaluator$meat_12)
  meat_22 <- as.numeric(basis %*% evaluator$meat_22)
  v11 <- meat_11
  v12 <- meat_12
  v22 <- meat_22
  coefficient <- t(vapply(seq_len(nrow(grid)), function(index) {
    evaluator$coefficient_intercept -
      evaluator$coefficient_slope %*% c(grid$beta[[index]], grid$theta[[index]])
  }, numeric(2L)))
  determinant <- v11 * v22 - v12^2
  scale <- pmax(abs(v11 * v22), abs(v12^2), 1)
  valid <- is.finite(determinant) & determinant > .Machine$double.eps * scale &
    is.finite(v11) & is.finite(v22) & v11 > 0 & v22 > 0
  q_statistic <- rep(Inf, nrow(grid))
  q_statistic[valid] <- (
    coefficient[valid, 1L]^2 * v22[valid] -
      2 * coefficient[valid, 1L] * coefficient[valid, 2L] * v12[valid] +
      coefficient[valid, 2L]^2 * v11[valid]
  ) / determinant[valid]
  q_statistic <- pmax(q_statistic, 0)
  f_statistic <- evaluator$delta * q_statistic / 2
  p_value <- stats::pf(
    f_statistic,
    df1 = 2,
    df2 = evaluator$df_denom,
    lower.tail = FALSE
  )
  grid$p_value <- as.numeric(p_value)
  grid
}

project_ar_bounds <- function(accepted, points) {
  if (!nrow(accepted)) {
    return(list(
      beta_low = NA_real_, beta_high = NA_real_,
      theta_low = NA_real_, theta_high = NA_real_,
      beta_lower_boundary = FALSE, beta_upper_boundary = FALSE,
      theta_lower_boundary = FALSE, theta_upper_boundary = FALSE,
      touches_boundary = FALSE
    ))
  }
  boundary <- c(
    beta_lower = any(accepted$beta_index == 1L),
    beta_upper = any(accepted$beta_index == points),
    theta_lower = any(accepted$theta_index == 1L),
    theta_upper = any(accepted$theta_index == points)
  )
  list(
    beta_low = if (boundary[["beta_lower"]]) NA_real_ else min(accepted$beta),
    beta_high = if (boundary[["beta_upper"]]) NA_real_ else max(accepted$beta),
    theta_low = if (boundary[["theta_lower"]]) NA_real_ else min(accepted$theta),
    theta_high = if (boundary[["theta_upper"]]) NA_real_ else max(accepted$theta),
    beta_lower_boundary = unname(boundary[["beta_lower"]]),
    beta_upper_boundary = unname(boundary[["beta_upper"]]),
    theta_lower_boundary = unname(boundary[["theta_lower"]]),
    theta_upper_boundary = unname(boundary[["theta_upper"]]),
    touches_boundary = any(boundary)
  )
}

accepted_touches_boundary <- function(accepted, points) {
  project_ar_bounds(accepted, points)$touches_boundary
}

accepted_component_count <- function(accepted, points) {
  if (!nrow(accepted)) {
    return(0L)
  }
  occupied <- matrix(FALSE, nrow = points, ncol = points)
  occupied[cbind(accepted$beta_index, accepted$theta_index)] <- TRUE
  visited <- matrix(FALSE, nrow = points, ncol = points)
  components <- 0L
  starts <- which(occupied, arr.ind = TRUE)
  for (start_index in seq_len(nrow(starts))) {
    start <- starts[start_index, ]
    if (visited[start[[1L]], start[[2L]]]) {
      next
    }
    components <- components + 1L
    queue_i <- start[[1L]]
    queue_j <- start[[2L]]
    head <- 1L
    while (head <= length(queue_i)) {
      i <- queue_i[[head]]
      j <- queue_j[[head]]
      head <- head + 1L
      if (visited[i, j]) {
        next
      }
      visited[i, j] <- TRUE
      neighbors <- rbind(
        c(i - 1L, j), c(i + 1L, j), c(i, j - 1L), c(i, j + 1L)
      )
      valid <- neighbors[, 1L] >= 1L & neighbors[, 1L] <= points &
        neighbors[, 2L] >= 1L & neighbors[, 2L] <= points
      neighbors <- neighbors[valid, , drop = FALSE]
      for (neighbor in seq_len(nrow(neighbors))) {
        ni <- neighbors[neighbor, 1L]
        nj <- neighbors[neighbor, 2L]
        if (occupied[ni, nj] && !visited[ni, nj]) {
          queue_i <- c(queue_i, ni)
          queue_j <- c(queue_j, nj)
        }
      }
    }
  }
  components
}

conventional_grid_point_accepted <- function(accepted, points) {
  if (!nrow(accepted)) {
    return(FALSE)
  }
  center_index <- as.integer((as.integer(points) + 1L) / 2L)
  any(
    accepted$beta_index == center_index &
      accepted$theta_index == center_index
  )
}

compute_ar_region <- function(d, conventional_fit, cell, spec) {
  validate_ar_spec(spec)
  if (is.null(conventional_fit$coefficients) || is.null(conventional_fit$std_error)) {
    stop("AR region requires a conventional two-parameter IV fit")
  }
  terms <- c("gimc_a", "gimc_gad_a")
  center <- as.numeric(conventional_fit$coefficients[terms])
  standard_error <- as.numeric(conventional_fit$std_error[terms])
  if (any(!is.finite(center)) || any(!is.finite(standard_error)) ||
    any(standard_error <= 0)) {
    stop("AR grid center or scale is invalid")
  }
  evaluator <- prepare_ar_evaluator(d)
  points <- as.integer(spec$ar_grid_points_per_axis)
  half_width <- as.numeric(spec$ar_initial_se_span) * standard_error
  alpha <- as.numeric(spec$alpha)
  max_expansions <- as.integer(spec$ar_max_expansions)
  expansion <- 0L
  accepted <- data.frame()
  evaluated <- NULL
  touches <- FALSE
  repeat {
    beta_grid <- seq(center[[1L]] - half_width[[1L]],
      center[[1L]] + half_width[[1L]], length.out = points
    )
    theta_grid <- seq(center[[2L]] - half_width[[2L]],
      center[[2L]] + half_width[[2L]], length.out = points
    )
    evaluated <- evaluate_ar_grid(evaluator, beta_grid, theta_grid)
    evaluated$beta_index <- match(evaluated$beta, beta_grid)
    evaluated$theta_index <- match(evaluated$theta, theta_grid)
    accepted <- evaluated[
      is.finite(evaluated$p_value) & evaluated$p_value >= alpha,
      c("beta", "theta", "beta_index", "theta_index", "p_value"),
      drop = FALSE
    ]
    if (!nrow(accepted)) {
      break
    }
    touches <- accepted_touches_boundary(accepted, points)
    if (!touches || expansion >= max_expansions) {
      break
    }
    half_width <- half_width * 2
    expansion <- expansion + 1L
  }
  components <- accepted_component_count(accepted, points)
  projection <- project_ar_bounds(accepted, points)
  status <- if (!nrow(accepted)) {
    "empty"
  } else if (touches && expansion >= max_expansions) {
    "unbounded"
  } else if (components > 1L) {
    "disjoint"
  } else {
    "bounded"
  }
  accepted_for_hash <- accepted[order(accepted$beta, accepted$theta),
    c("beta", "theta"), drop = FALSE
  ]
  list(
    analysis_family = as.character(cell$analysis_family),
    outcome_id = as.character(cell$outcome_id),
    horizon = as.integer(cell$horizon),
    gad_version = as.character(cell$gad_version),
    sample_version = as.character(cell$sample_version),
    status = status,
    beta_conf_low = projection$beta_low,
    beta_conf_high = projection$beta_high,
    theta_conf_low = projection$theta_low,
    theta_conf_high = projection$theta_high,
    expansions = as.integer(expansion),
    grid_points_per_axis = points,
    accepted_points = as.integer(nrow(accepted)),
    conventional_point_accepted = conventional_grid_point_accepted(
      accepted, points
    ),
    connected_components = as.integer(components),
    accepted_hash = canonical_frame_sha256(accepted_for_hash),
    alpha = alpha,
    inference_method = "anderson_rubin_cr2_htz",
    reference_df = as.numeric(evaluator$df_denom),
    clusters = as.integer(evaluator$clusters),
    n = as.integer(evaluator$observations)
  )
}
