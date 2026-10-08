canonical_list_sha256 <- function(value) {
  text <- jsonlite::toJSON(
    value,
    auto_unbox = TRUE,
    null = "null",
    digits = 17,
    pretty = FALSE,
    dataframe = "rows"
  )
  digest::digest(enc2utf8(text), algo = "sha256", serialize = FALSE)
}

canonical_frame_sha256 <- function(d) {
  if (!is.data.frame(d)) {
    stop("canonical frame must be a data frame")
  }
  if (nrow(d) == 0L) {
    return(canonical_list_sha256(list()))
  }
  ordered <- d[
    do.call(order, c(d[names(d)], list(na.last = TRUE))),
    names(d),
    drop = FALSE
  ]
  rownames(ordered) <- NULL
  canonical_list_sha256(unname(split(ordered, seq_len(nrow(ordered)))))
}

ensure_parent_directory <- function(path) {
  parent <- dirname(path)
  if (!dir.exists(parent) && !dir.create(parent, recursive = TRUE)) {
    stop("cannot create output directory: ", parent)
  }
  invisible(parent)
}

atomic_write_json <- function(value, path) {
  ensure_parent_directory(path)
  partial <- paste0(path, ".partial")
  if (file.exists(partial)) {
    unlink(partial)
  }
  on.exit(if (file.exists(partial)) unlink(partial), add = TRUE)
  jsonlite::write_json(
    value,
    partial,
    auto_unbox = TRUE,
    null = "null",
    digits = 17,
    pretty = TRUE
  )
  if (!file.rename(partial, path)) {
    stop("atomic JSON rename failed")
  }
  invisible(path)
}

atomic_write_csv <- function(value, path) {
  ensure_parent_directory(path)
  partial <- paste0(path, ".partial")
  if (file.exists(partial)) {
    unlink(partial)
  }
  on.exit(if (file.exists(partial)) unlink(partial), add = TRUE)
  serialized <- value
  double_columns <- names(serialized)[vapply(
    serialized, is.double, logical(1L)
  )]
  for (column in double_columns) {
    formatted <- sprintf("%.17g", serialized[[column]])
    formatted[is.na(serialized[[column]])] <- NA_character_
    serialized[[column]] <- formatted
  }
  utils::write.csv(serialized, partial, row.names = FALSE, na = "")
  if (!file.rename(partial, path)) {
    stop("atomic CSV rename failed")
  }
  invisible(path)
}

require_scalar_string <- function(value, field) {
  if (!is.character(value) || length(value) != 1L || is.na(value) || !nzchar(value)) {
    stop(field, " must be one non-empty string")
  }
  value
}

validate_run_context <- function(run_context) {
  required <- c(
    "run_id", "spec_id", "input_authority_hash", "git_commit",
    "renv_lock_sha256", "evidence_policy_sha256", "seed", "created_at_utc"
  )
  missing <- setdiff(required, names(run_context))
  if (length(missing)) {
    stop("run context is missing: ", paste(missing, collapse = ", "))
  }
  for (field in setdiff(required, "seed")) {
    require_scalar_string(run_context[[field]], paste0("run_context.", field))
  }
  if (
    length(run_context$seed) != 1L || is.na(run_context$seed) ||
      !is.numeric(run_context$seed) || run_context$seed <= 0
  ) {
    stop("run_context.seed must be one positive integer")
  }
  invisible(run_context)
}

FROZEN_SSC_CONFIG <- paste0(
  "fixest:K.adj=TRUE;K.fixef=nonnested;K.exact=FALSE;",
  "G.adj=TRUE;G.df=min;t.df=min"
)

FROZEN_PACKAGE_VERSIONS_JSON <- paste0(
  '{"R.JuliaConnectoR":"1.1.5","R.arrow":"25.0.1",',
  '"R.clubSandwich":"0.7.0",',
  '"R.data.table":"1.18.6.1","R.digest":"0.6.39",',
  '"R.dqrng":"0.4.1","R.fixest":"0.14.2",',
  '"R.fwildclusterboot":"0.14.3","R.ggplot2":"4.0.3",',
  '"R.ivreg":"0.6.8","R.jsonlite":"2.0.0",',
  '"julia.StableRNGs":"1.0.4",',
  '"julia.WildBootTests":"0.9.8","python.numpy":"2.4.6",',
  '"python.polars":"1.42.1","python.scipy":"1.18.0"}'
)

frozen_fixest_ssc <- function() {
  fixest::ssc(
    K.adj = TRUE,
    K.fixef = "nonnested",
    K.exact = FALSE,
    G.adj = TRUE,
    G.df = "min",
    t.df = "min"
  )
}

cluster_reference_df <- function(clusters) {
  clusters <- as.integer(clusters)
  if (length(clusters) != 1L || is.na(clusters) || clusters < 2L) {
    stop("cluster t reference requires at least two clusters")
  }
  as.numeric(clusters - 1L)
}

cluster_t_p_value <- function(estimate, std_error, reference_df) {
  if (
    length(std_error) != 1L || !is.finite(std_error) || std_error <= 0 ||
      length(reference_df) != 1L || !is.finite(reference_df) || reference_df <= 0
  ) {
    stop("cluster t inference requires positive standard error and df")
  }
  2 * stats::pt(-abs(estimate / std_error), df = reference_df)
}

cluster_t_bounds <- function(estimate, std_error, reference_df) {
  critical <- stats::qt(0.975, df = reference_df)
  c(low = estimate - critical * std_error, high = estimate + critical * std_error)
}

frozen_result_metadata <- function(
  run_context, d, clusters,
  reference_distribution = "cluster_t",
  reference_df = cluster_reference_df(clusters),
  ssc_config = FROZEN_SSC_CONFIG
) {
  validate_run_context(run_context)
  if (!is.data.frame(d) || !nrow(d) || !"treatment_time" %in% names(d)) {
    stop("result metadata requires a non-empty treatment_time frame")
  }
  years <- as.integer(d$treatment_time)
  if (anyNA(years) || any(!is.finite(years))) {
    stop("result metadata year range is invalid")
  }
  list(
    evidence_policy_sha256 = as.character(run_context$evidence_policy_sha256),
    year_min = as.integer(min(years)),
    year_max = as.integer(max(years)),
    r_version = "4.6.1",
    python_version = "3.12.13",
    julia_version = "1.12.7",
    package_versions_json = FROZEN_PACKAGE_VERSIONS_JSON,
    random_seed = as.numeric(run_context$seed),
    ssc_config = as.character(ssc_config),
    reference_distribution = as.character(reference_distribution),
    reference_df = as.numeric(reference_df)
  )
}

matrix_rank <- function(value) {
  singular_values <- svd(value, nu = 0L, nv = 0L)$d
  if (!length(singular_values) || max(singular_values) == 0) {
    return(0L)
  }
  tolerance <- max(dim(value)) * .Machine$double.eps * max(singular_values)
  as.integer(sum(singular_values > tolerance))
}
