#!/usr/bin/env Rscript
# rfm_data_prep.R  ----------------------------------------------------------
#  Build clean RFM panels from raw CSVs  (UCI & CDNOW)
#  Run:  Rscript rfm_data_prep.R      (non-interactive, silent)
#        source("rfm_data_prep.R")    (interactive, plots shown)
# ---------------------------------------------------------------------------

# ---- 0.  one-time package loader -----------------------------------------
pkg_vec <- c("tidyverse", "lubridate", "readr", "here", "janitor", "mgcv",
             "ggplot2", "scales", "dplyr", "tidyr")
for (p in pkg_vec)
  if (!requireNamespace(p, quietly = TRUE)) install.packages(p, repos = "https://cloud.r-project.org")
suppressPackageStartupMessages(library(tidyverse))

# ---- 1.  ingest functions -------------------------------------------------
ingest_uci <- function(path) {
  raw <- readr::read_csv(path, show_col_types = FALSE) %>%          # silent
    mutate(
      date         = lubridate::dmy_hm(InvoiceDate),
      Monetary     = Quantity * UnitPrice,
      customer_id  = as.character(CustomerID)
    ) %>%
    filter(!is.na(customer_id), Quantity > 0, UnitPrice > 0,
           !stringr::str_detect(InvoiceNo, "^C")) %>%
    group_by(customer_id, WeekStart = lubridate::floor_date(date, "week")) %>%
    summarise(WeeklySpend = sum(Monetary), n_transactions = n_distinct(InvoiceNo),
              .groups = "drop")
  return(raw)
}

ingest_cdnow <- function(path) {
  raw <- readr::read_csv(path, show_col_types = FALSE) %>%
    transmute(
      customer_id = as.character(customer),
      date        = lubridate::ymd(date),               # original col is YYYY-MM-DD
      Monetary    = dollar,
      WeekStart   = lubridate::floor_date(date, "week")
    ) %>%
    group_by(customer_id, WeekStart) %>%
    summarise(WeeklySpend = sum(Monetary), n_transactions = n(),
              .groups = "drop")
  return(raw)
}

# ---- 2.  baseline RFM feature builder -----------------------------------
build_rfm_baseline <- function(weekly_df) {
  # weekly_df cols: customer_id, WeekStart, WeeklySpend, n_transactions
  min_week <- min(weekly_df$WeekStart)
  max_week <- max(weekly_df$WeekStart)
  full_grid <- expand.grid(customer_id = unique(weekly_df$customer_id),
                           WeekStart   = seq.Date(min_week, max_week, by = 7)) %>%
    as_tibble()

  rfm <- weekly_df %>%
    right_join(full_grid, by = c("customer_id", "WeekStart")) %>%
    mutate(
      WeeklySpend    = replace_na(WeeklySpend, 0),
      n_transactions = replace_na(n_transactions, 0)
    ) %>%
    group_by(customer_id) %>%
    mutate(
      R_weeks = as.numeric(max_week - WeekStart) / 7,        # recency
      F_run   = cumsum(n_transactions),                       # cumulative freq
      M_run   = cummean(WeeklySpend),                         # cumulative mean spend
      zero_incidence_run = cumsum(WeeklySpend == 0) / row_number()
    ) %>%
    ungroup() %>%
    arrange(customer_id, WeekStart)

  return(rfm)
}

# ---- 3.  plotting helpers (silent in batch) -----------------------------
plot_zero_incidence <- function(rfm, set_name) {
  p <- ggplot(rfm, aes(x = WeekStart, y = zero_incidence_run, group = customer_id)) +
    geom_line(alpha = .1) + stat_summary(fun = mean, geom = "line", colour = "red", size = 1) +
    labs(title = paste(set_name, "- zero-incidence paths"),
         y = "Running zero-incidence rate", x = "Week") +
    theme_minimal()
  print(p)
}

# ---- 4.  end-to-end wrappers -------------------------------------------
build_rfm_uci  <- function(path_raw) build_rfm_baseline(ingest_uci(path_raw))
build_rfm_cdnow<- function(path_raw) build_rfm_baseline(ingest_cdnow(path_raw))

# ---- 5.  reproducible build (only runs when script is *executed*) --------
if (sys.nframe() == 0) {          # true when Rscript, false when source()d inside another session
  here::i_am("rfm_data_prep.R")   # set repo root once
  cat("Building UCI panel...\n")
  uci_rfm  <- build_rfm_uci (here::here("data", "Online Retail.csv"))
  saveRDS(uci_rfm,  here::here("data", "uci_rfm.rds"))
  cat("Building CDNOW panel...\n")
  cdnow_rfm<- build_rfm_cdnow(here::here("data", "cdnow.csv"))
  saveRDS(cdnow_rfm, here::here("data", "cdnow_rfm.rds"))
  cat("Panels saved to data/*.rds\n")
}

# ---- 6.  interactive extras (plots) -------------------------------------
if (interactive()) {
  print("Showing diagnostic plots...")
  plot_zero_incidence(uci_rfm,  "UCI")
  plot_zero_incidence(cdnow_rfm,"CDNOW")
}
