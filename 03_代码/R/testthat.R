test_root <- file.path("03_代码", "tests_r", "testthat")
if (dir.exists(test_root)) {
  if (!requireNamespace("testthat", quietly = TRUE)) {
    stop("testthat is not installed")
  }
  Sys.setenv(GREEN_DEBT_PROJECT_ROOT = normalizePath(getwd(), mustWork = TRUE))
  Sys.setenv(
    JULIACONNECTOR_JULIAOPTS = paste0(
      "--project=",
      file.path(
        normalizePath(getwd(), mustWork = TRUE),
        "julia"
      )
    )
  )
  testthat::test_dir(test_root, reporter = "summary")
} else {
  cat("No R analysis tests are registered yet.\n")
}
