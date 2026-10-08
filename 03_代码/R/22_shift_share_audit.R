shift_share_sample_columns <- c(
  "economy_id", "treatment_time", "raw_Z", "raw_Z_GAD",
  "raw_gad_lag", "z_a", "z_gad_a"
)

normalize_shock_keys <- function(shocks) {
  normalized <- shocks
  if (!"economy_id" %in% names(normalized) && "importer" %in% names(normalized)) {
    normalized$economy_id <- as.character(normalized$importer)
  }
  if (!"treatment_time" %in% names(normalized) && "year" %in% names(normalized)) {
    normalized$treatment_time <- as.integer(normalized$year)
  }
  required <- c(
    "economy_id", "treatment_time", "exporter", "hs6", "year", "contribution"
  )
  missing <- setdiff(required, names(normalized))
  if (length(missing)) {
    stop("shift-share shocks are missing: ", paste(missing, collapse = ", "))
  }
  normalized
}

validate_shift_share_sample <- function(sample) {
  if (!is.data.frame(sample) || !nrow(sample)) {
    stop("shift-share model sample must be non-empty")
  }
  missing <- setdiff(shift_share_sample_columns, names(sample))
  if (length(missing)) {
    stop("shift-share sample is missing: ", paste(missing, collapse = ", "))
  }
  if (anyDuplicated(paste(sample$economy_id, sample$treatment_time))) {
    stop("shift-share sample has duplicate economy-time keys")
  }
  numeric <- setdiff(shift_share_sample_columns, "economy_id")
  if (anyNA(sample[shift_share_sample_columns]) ||
    any(!is.finite(as.matrix(sample[numeric])))) {
    stop("shift-share sample contains missing or nonfinite authority values")
  }
  invisible(sample)
}

decompose_main_instruments <- function(sample, shocks, tolerance = 1e-12) {
  validate_shift_share_sample(sample)
  shocks <- normalize_shock_keys(shocks)
  shock_value_columns <- c(
    "economy_id", "treatment_time", "exporter", "hs6", "year", "contribution"
  )
  if (
    anyNA(shocks[shock_value_columns]) ||
      any(!is.finite(as.numeric(shocks$contribution)))
  ) {
    stop("shift-share shocks contain missing or nonfinite contributions")
  }
  if (any(sample$raw_Z == 0 & abs(sample$z_a) > tolerance)) {
    stop("cannot reconstruct clipped Z from zero raw Z")
  }
  if (any(sample$raw_Z_GAD == 0 & abs(sample$z_gad_a) > tolerance)) {
    stop("cannot reconstruct clipped Z_GAD from zero raw Z_GAD")
  }
  pieces <- merge(
    shocks,
    sample,
    by = c("economy_id", "treatment_time"),
    all = FALSE,
    sort = FALSE
  )
  if (!nrow(pieces)) {
    stop("shift-share shocks do not overlap the exact model sample")
  }
  pieces$z_piece <- ifelse(
    pieces$raw_Z == 0,
    0,
    pieces$contribution * pieces$z_a / pieces$raw_Z
  )
  pieces$z_gad_piece <- ifelse(
    pieces$raw_Z_GAD == 0,
    0,
    pieces$contribution * pieces$raw_gad_lag *
      pieces$z_gad_a / pieces$raw_Z_GAD
  )
  pieces$shock_id <- paste(
    pieces$exporter, pieces$hs6, pieces$year, sep = "|"
  )
  pieces$shock_cluster_id <- paste(pieces$exporter, pieces$hs6, sep = "|")

  reconstructed <- aggregate(
    cbind(z_piece, z_gad_piece) ~ economy_id + treatment_time,
    pieces,
    sum
  )
  checked <- merge(
    sample[c("economy_id", "treatment_time", "z_a", "z_gad_a")],
    reconstructed,
    by = c("economy_id", "treatment_time"),
    all.x = TRUE,
    sort = FALSE
  )
  checked$z_piece[is.na(checked$z_piece)] <- 0
  checked$z_gad_piece[is.na(checked$z_gad_piece)] <- 0
  z_error <- max(abs(checked$z_piece - checked$z_a))
  z_gad_error <- max(abs(checked$z_gad_piece - checked$z_gad_a))
  if (z_error > tolerance || z_gad_error > tolerance) {
    stop(
      "scaled shock contributions fail exact instrument reconstruction: ",
      format(max(z_error, z_gad_error), digits = 17)
    )
  }
  attr(pieces, "z_reconstruction_error") <- z_error
  attr(pieces, "z_gad_reconstruction_error") <- z_gad_error
  pieces <- pieces[order(
    pieces$shock_cluster_id, pieces$year,
    pieces$economy_id, pieces$treatment_time
  ), , drop = FALSE]
  rownames(pieces) <- NULL
  pieces
}

shock_cross_moment <- function(piece) {
  z <- as.matrix(piece[c("z_piece", "z_gad_piece")])
  x <- as.matrix(piece[c("gimc_a", "gimc_gad_a")])
  crossprod(z, x)
}

ranked_cross_moment_inverse <- function(q) {
  if (!is.matrix(q) || nrow(q) != ncol(q) || any(!is.finite(q))) {
    stop("shock cross-moment matrix must be finite and square")
  }
  rank <- matrix_rank(q)
  inverse <- NULL
  if (rank == nrow(q)) {
    inverse <- tryCatch(solve(q), error = function(error) NULL)
    if (!is.null(inverse) && any(!is.finite(inverse))) {
      inverse <- NULL
    }
  }
  list(rank = rank, inverse = inverse)
}

empty_shift_share_audit <- function(pieces, q, status, rank = matrix_rank(q)) {
  list(
    summary = list(
      shock_observations = as.integer(length(unique(pieces$shock_id))),
      shock_clusters = as.integer(length(unique(pieces$shock_cluster_id))),
      signed_weight_sum = NA_real_,
      absolute_weight_sum = NA_real_,
      hhi_absolute = NA_real_,
      top1_absolute_share = NA_real_,
      top5_absolute_share = NA_real_,
      negative_weight_share = NA_real_,
      z_reconstruction_error = as.numeric(
        attr(pieces, "z_reconstruction_error")
      ),
      z_gad_reconstruction_error = as.numeric(
        attr(pieces, "z_gad_reconstruction_error")
      ),
      shock_std_error_gimc = NA_real_,
      shock_std_error_interaction = NA_real_,
      shock_inference_status = status,
      cross_moment_rank = as.integer(rank)
    ),
    weights = data.frame(),
    covariance = matrix(NA_real_, 2L, 2L)
  )
}

generalized_rotemberg <- function(fixture, tolerance = 1e-10) {
  if (!is.list(fixture) || !all(c("sample", "shocks") %in% names(fixture))) {
    stop("generalized Rotemberg audit requires sample and shocks")
  }
  required_model <- c("gimc_a", "gimc_gad_a", "residual")
  missing_model <- setdiff(required_model, names(fixture$sample))
  if (length(missing_model)) {
    stop(
      "shift-share audit sample is missing: ",
      paste(missing_model, collapse = ", ")
    )
  }
  pieces <- decompose_main_instruments(
    fixture$sample, fixture$shocks, tolerance = 1e-12
  )
  z <- as.matrix(pieces[c("z_piece", "z_gad_piece")])
  x <- as.matrix(pieces[c("gimc_a", "gimc_gad_a")])
  q <- crossprod(z, x)
  ranked_inverse <- ranked_cross_moment_inverse(q)
  q_inverse <- ranked_inverse$inverse
  if (is.null(q_inverse)) {
    return(empty_shift_share_audit(
      pieces, q, "unavailable", rank = ranked_inverse$rank
    ))
  }

  calculation <- data.table::as.data.table(pieces)
  calculation[, `:=`(
    score_gimc = z_piece * residual,
    score_interaction = z_gad_piece * residual,
    q11 = z_piece * gimc_a,
    q12 = z_piece * gimc_gad_a,
    q21 = z_gad_piece * gimc_a,
    q22 = z_gad_piece * gimc_gad_a
  )]
  weights <- calculation[, .(
    exporter = exporter[[1L]],
    hs6 = hs6[[1L]],
    year = as.integer(year[[1L]]),
    shock_cluster_id = shock_cluster_id[[1L]],
    q11 = sum(q11), q12 = sum(q12),
    q21 = sum(q21), q22 = sum(q22)
  ), by = shock_id]
  weights[, signed_weight := (
    q_inverse[1L, 1L] * q11 + q_inverse[1L, 2L] * q21 +
      q_inverse[2L, 1L] * q12 + q_inverse[2L, 2L] * q22
  ) / 2]
  weights[, absolute_weight := abs(signed_weight)]
  weights <- as.data.frame(weights[
    , c(
      "exporter", "hs6", "year", "shock_id", "shock_cluster_id",
      "signed_weight", "absolute_weight"
    ),
    with = FALSE
  ])
  signed_sum <- sum(weights$signed_weight)
  if (!is.finite(signed_sum) || abs(signed_sum - 1) > tolerance) {
    stop("generalized Rotemberg signed weights do not sum to one")
  }
  absolute_sum <- sum(weights$absolute_weight)
  if (!is.finite(absolute_sum) || absolute_sum <= 0) {
    stop("generalized Rotemberg absolute weights are invalid")
  }
  weights <- weights[order(
    -weights$absolute_weight, weights$shock_id
  ), , drop = FALSE]
  weights$absolute_rank <- seq_len(nrow(weights))
  rownames(weights) <- NULL
  absolute_share <- weights$absolute_weight / absolute_sum

  cluster_scores <- calculation[, .(
    score_gimc = sum(score_gimc),
    score_interaction = sum(score_interaction)
  ), by = shock_cluster_id]
  covariance <- matrix(NA_real_, nrow = 2L, ncol = 2L)
  standard_error <- c(NA_real_, NA_real_)
  inference_status <- "unavailable"
  if (nrow(cluster_scores) >= 2L) {
    score_matrix <- as.matrix(cluster_scores[
      , c("score_gimc", "score_interaction"), with = FALSE
    ])
    meat <- crossprod(score_matrix)
    candidate_covariance <- q_inverse %*% meat %*% t(q_inverse)
    diagonal <- diag(candidate_covariance)
    if (
      all(is.finite(candidate_covariance)) &&
        all(diagonal >= -tolerance)
    ) {
      covariance <- candidate_covariance
      standard_error <- sqrt(pmax(diagonal, 0))
      inference_status <- "available"
    }
  }
  summary <- list(
    shock_observations = as.integer(nrow(weights)),
    shock_clusters = as.integer(nrow(cluster_scores)),
    signed_weight_sum = as.numeric(signed_sum),
    absolute_weight_sum = as.numeric(absolute_sum),
    hhi_absolute = as.numeric(sum(absolute_share^2)),
    top1_absolute_share = as.numeric(sum(head(absolute_share, 1L))),
    top5_absolute_share = as.numeric(sum(head(absolute_share, 5L))),
    negative_weight_share = as.numeric(sum(
      weights$absolute_weight[weights$signed_weight < 0]
    ) / absolute_sum),
    z_reconstruction_error = as.numeric(
      attr(pieces, "z_reconstruction_error")
    ),
    z_gad_reconstruction_error = as.numeric(
      attr(pieces, "z_gad_reconstruction_error")
    ),
    shock_std_error_gimc = as.numeric(standard_error[[1L]]),
    shock_std_error_interaction = as.numeric(standard_error[[2L]]),
    shock_inference_status = inference_status,
    cross_moment_rank = as.integer(ranked_inverse$rank)
  )
  list(summary = summary, weights = weights, covariance = covariance)
}

exact_shift_share_sample <- function(panel, model_keys) {
  identity <- c(
    "economy_id", "treatment_time", "outcome_id", "horizon",
    "gad_version", "sample_version"
  )
  authority <- c(identity, "Z", "Z_GAD", "gad_lag", "Z_p01_p99", "Z_GAD_p01_p99")
  missing_panel <- setdiff(authority, names(panel))
  missing_keys <- setdiff(identity, names(model_keys))
  if (length(missing_panel) || length(missing_keys)) {
    stop("exact shift-share sample is missing frozen identity or authority columns")
  }
  joined <- merge(
    model_keys,
    panel[authority],
    by = identity,
    all.x = TRUE,
    sort = FALSE
  )
  names(joined)[match(
    c("Z", "Z_GAD", "gad_lag", "Z_p01_p99", "Z_GAD_p01_p99"),
    names(joined)
  )] <- c("raw_Z", "raw_Z_GAD", "raw_gad_lag", "z_a", "z_gad_a")
  if (nrow(joined) != nrow(model_keys) || anyNA(joined[c(
    "raw_Z", "raw_Z_GAD", "raw_gad_lag", "z_a", "z_gad_a"
  )])) {
    stop("exact shift-share authority join is incomplete")
  }
  joined
}
