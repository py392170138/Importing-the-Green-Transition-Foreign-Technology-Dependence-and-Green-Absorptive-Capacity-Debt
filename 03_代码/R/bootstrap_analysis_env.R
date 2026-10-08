if (FALSE) {
  arrow::read_parquet
  data.table::data.table
  fixest::feols
  ivreg::ivreg
  clubSandwich::vcovCR
  fwildclusterboot::boottest
  dqrng::dqset.seed
  ggplot2::ggplot
  ragg::agg_png
  svglite::svglite
  jsonlite::toJSON
  digest::digest
  modelsummary::modelsummary
  testthat::test_dir
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L || !(args[[1L]] %in% c("initialize", "restore", "verify"))) {
  stop("expected exactly one action: initialize, restore, or verify")
}
action <- args[[1L]]
root <- normalizePath(getwd(), mustWork = TRUE)

proxy_names <- c(
  "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
  "http_proxy", "https_proxy", "all_proxy"
)
present_proxies <- proxy_names[nzchar(Sys.getenv(proxy_names, unset = ""))]
if (length(present_proxies) > 0L) {
  stop(sprintf("proxy variables reached R child: %s", paste(present_proxies, collapse = ", ")))
}
if (!identical(Sys.getenv("NO_PROXY"), "*") || !identical(Sys.getenv("no_proxy"), "*")) {
  stop("NO_PROXY and no_proxy must both be wildcarded")
}

bootstrap_lib <- file.path(root, ".r-bootstrap-library")
dir.create(bootstrap_lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(bootstrap_lib, .libPaths()))
options(repos = c(CRAN = "https://cloud.r-project.org"))

packages <- c(
  "arrow", "data.table", "fixest", "ivreg", "clubSandwich",
  "fwildclusterboot", "dqrng", "ggplot2", "ragg", "svglite",
  "jsonlite", "digest",
  "modelsummary", "testthat"
)
fwildclusterboot_commit <- "336bb574eba169ac0183317f01d0564791d8122f"

if (!requireNamespace("renv", quietly = TRUE)) {
  if (identical(action, "verify")) stop("renv is not installed")
  install.packages("renv", lib = bootstrap_lib)
}

if (identical(action, "initialize")) {
  renv::init(project = root, bare = TRUE, restart = FALSE)
  renv::install(
    packages[packages != "fwildclusterboot"],
    project = root
  )
  renv::install("summclust@0.7.2", project = root)
  renv::install(
    paste0("s3alfisc/fwildclusterboot@", fwildclusterboot_commit),
    project = root
  )
  renv::snapshot(
    project = root,
    packages = c("renv", packages),
    prompt = FALSE
  )
} else if (identical(action, "restore")) {
  lockfile <- file.path(root, "renv.lock")
  if (!file.exists(lockfile)) stop("renv.lock is missing")
  renv::restore(project = root, prompt = FALSE)
} else {
  if (!identical(Sys.getenv("RENV_CONFIG_OFFLINE"), "TRUE")) {
    stop("verification must run with RENV_CONFIG_OFFLINE=TRUE")
  }
  renv::load(project = root)
  status <- renv::status(project = root)
  if (!isTRUE(status$synchronized)) {
    stop("renv library and lockfile are not synchronized")
  }
}
