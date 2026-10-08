args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("usage: extract_ilostat_rds.R INPUT.rds OUTPUT.csv", call. = FALSE)
}

input_path <- args[[1L]]
output_path <- args[[2L]]
value <- readRDS(input_path)
if (!is.data.frame(value)) {
  stop("ILOSTAT RDS object is not a data frame", call. = FALSE)
}

normalized <- as.data.frame(
  lapply(value, function(column) {
    if (is.factor(column)) as.character(column) else column
  }),
  check.names = FALSE,
  stringsAsFactors = FALSE
)
write.table(
  normalized,
  file = output_path,
  sep = ",",
  row.names = FALSE,
  col.names = TRUE,
  quote = TRUE,
  qmethod = "double",
  na = "",
  fileEncoding = "UTF-8"
)
