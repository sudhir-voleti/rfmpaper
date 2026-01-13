#!/usr/bin/env Rscript
# rfm_data_prep.R  ----------------------------------------------------------
#  * ingest raw CSV  (UCI / CDNOW)
#  * build full RFM panel
#  * export .rds ready for modelling
#  * optional diagnostics plots (interactive only)
# ---------------------------------------------------------------------------

# ---- 0.  auto-install any missing packages ---------------------------------
req_pkg <- c("tidyverse", "lubridate", "readxl", "curl", "janitor",
             "ggplot2", "scales", "dplyr", "tidyr", "ggrepel", "readr",
             "ggraph", "tidygraph", "openssl")   # openssl only for SHA if you keep it

for (p in req_pkg)
  if (!requireNamespace(p, quietly = TRUE))
    install.packages(p, repos = "https://cloud.r-project.org")
suppressPackageStartupMessages(library(tidyverse))

# ===== 1.  define functions ================================================
ingest_uci <- function(csv_path) {
  read_csv(csv_path, show_col_types = FALSE) %>%          # your local UCI csv
    mutate(
      date        = as.Date(lubridate::mdy_hm(InvoiceDate)),
      Monetary    = Quantity * UnitPrice,
      customer_id = as.character(CustomerID),
      WeekStart   = floor_date(date, "week")
    ) %>%
    filter(!is.na(customer_id), Quantity > 0, UnitPrice > 0,
           !str_detect(InvoiceNo, "^C")) %>%
    group_by(customer_id, WeekStart) %>%
    summarise(WeeklySpend = sum(Monetary), n_transactions = n_distinct(InvoiceNo),
              .groups = "drop")
}

ingest_cdnow <- function(csv_path) {
  read_csv(csv_path, show_col_types = FALSE) %>%          # your local CDNOW csv
    rename(customer_id = customer,
           date        = date,
           Monetary    = dollar) %>%
    mutate(date      = as.Date(date, format = "%Y%m%d"),
           WeekStart = floor_date(date, "week")) %>%
    group_by(customer_id, WeekStart) %>%
    summarise(WeeklySpend = sum(Monetary), n_transactions = n(),
              .groups = "drop")
}

build_rfm_baseline <- function(weekly_df) {
  min_w <- min(weekly_df$WeekStart)
  max_w <- max(weekly_df$WeekStart)
  grid  <- expand.grid(customer_id = unique(weekly_df$customer_id),
                       WeekStart   = seq.Date(min_w, max_w, by = 7)) %>% as_tibble()

  weekly_df %>%
    right_join(grid, by = c("customer_id", "WeekStart")) %>%
    mutate(
      WeeklySpend    = replace_na(WeeklySpend, 0),
      n_transactions = replace_na(n_transactions, 0)
    ) %>%
    group_by(customer_id) %>%
    arrange(WeekStart) %>%
    mutate(
      R_weeks            = as.numeric(max_w - WeekStart) / 7,
      F_run              = cumsum(n_transactions),
      M_run              = cummean(WeeklySpend),
      zero_incidence_run = cumsum(WeeklySpend == 0) / row_number()
    ) %>%
    ungroup()
}

# ===== 2.  point to your local files ========================================
path_uci <- "https://archive.ics.uci.edu/static/public/352/data.csv"
path_cdnow <- "/Users/sudhirvoleti/research related/HMM n tweedie in RFM Nov 2025/CDNOW/cdnow_raw_10pct.csv"               # <-- drop your cleaned CDNOW csv here

# ===== 3.  run UCI ==========================================================
uci_weekly <- ingest_uci(path_uci)
uci_feat   <- build_rfm_baseline(uci_weekly)
saveRDS(uci_feat, "uci_modelling.rds")
cat("UCI done -> uci_modelling.rds\n")

# ===== 4.  run CDNOW ========================================================
cdnow_weekly <- ingest_cdnow(path_cdnow)
cdnow_feat   <- build_rfm_baseline(cdnow_weekly)
saveRDS(cdnow_feat, "cdnow_modelling.rds")
cat("CDNOW done -> cdnow_modelling.rds\n")


# ---- 6.  interactive diagnostics (only when sourced in RStudio) -----------
if (interactive()) {
  print("Generating diagnostic plots...")
  plot_spend_surface(uci_rfm,  "UCI")
  plot_spend_surface(cdnow_rfm,"CDNOW")

  # (gamma_mat needs your HMM posterior – stub now, fill after SMC)
  # gamma_uci <- matrix(c(...), nrow = 3, ncol = 3)   # posterior mean
  # plot_trans_mat(gamma_uci, "UCI")
}
