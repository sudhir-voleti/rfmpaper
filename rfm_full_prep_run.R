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

# ---- 1.  UCI online retail -------------------------------------------------
ingest_uci <- function() {
  tmp  <- tempfile(fileext = ".csv")
  url  <- "https://archive.ics.uci.edu/static/public/352/data.csv"
  
  # curl with progress + resume if partial
  h <- curl::new_handle()
  curl::handle_setopt(h, ssl_verifyhost = 0, ssl_verifypeer = 0,
                      progressfunction = function(down, up, ultotal, ulnow) {
                        if (ultotal > 0)
                          message(sprintf("Downloaded %.1f / %.1f MB",
                                          down / 2^20, ultotal / 2^20), appendLF = FALSE)
                        TRUE
                      }, noprogress = 0)
  curl::curl_download(url, tmp, mode = "wb", handle = h)
  
  raw <- read_csv(tmp, show_col_types = FALSE) %>% janitor::clean_names()
  unlink(tmp)
  
  raw %>%
    mutate(
      date        = as.Date(lubridate::mdy_hm(invoice_date)),
      Monetary    = quantity * unit_price,
      customer_id = as.character(customer_id),
      WeekStart   = lubridate::floor_date(date, "week")
    ) %>%
    filter(!is.na(customer_id), quantity > 0, unit_price > 0,
           !str_detect(invoice_no, "^C")) %>%
    group_by(customer_id, WeekStart) %>%
    summarise(WeeklySpend = sum(Monetary), n_transactions = n_distinct(invoice_no),
              .groups = "drop") %>%
    return()
}

system.time ({  uci = ingest_uci() })  # 88s
cat("UCI sample customers:", n_distinct(uci$customer_id), "\n") 
glimpse(uci)

# =====  CDNOW 1/10 sample  –  direct from Hardie =================================
parse_customer_strings <- function(string_vector) {
  read.table(text = string_vector,
             header = FALSE,
             col.names = c("cust_id", "new_cust_id", "date", "qty", "spend"),
             colClasses = c("character", "character", "character", "integer", "numeric"))
} # helper func

ingest_cdnow_sample <- function() {
  tmp_zip <- tempfile(fileext = ".zip")
  curl::curl_download("http://www.brucehardie.com/datasets/CDNOW_sample.zip", tmp_zip)
  tmp_dir <- tempdir(); unzip(tmp_zip, exdir = tmp_dir)
  txt_path <- file.path(tmp_dir, "CDNOW_sample.txt")
  
  raw <- parse_customer_strings(readLines(txt_path)) %>%
    filter(complete.cases(cust_id, new_cust_id, date, qty, spend))
  
  unlink(tmp_zip, recursive = TRUE)
  
  raw %>%
    transmute(
      customer_id = as.character(new_cust_id),
      date        = as.Date(date, format = "%Y%m%d"),
      Monetary    = spend,
      WeekStart   = lubridate::floor_date(date, "week")
    ) %>%
    group_by(customer_id, WeekStart) %>%
    summarise(WeeklySpend = sum(Monetary), n_transactions = n(),
              .groups = "drop")
} # main func

system.time ({ cdnow <- ingest_cdnow_sample() })  # 4 s
cat("CDNOW sample customers:", n_distinct(cdnow$customer_id), "\n") 
glimpse(cdnow)

# ---- 3. Build RFM panels ---------------------------------------------
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

system.time({   uci_rfm = build_rfm_baseline(uci) }) # 0.15s
glimpse(uci_rfm) # 234k x 8

system.time({   cdnow_rfm = build_rfm_baseline(cdnow) }) # 0.15s
glimpse(cdnow_rfm) # 186k x 8

# ===== Table 2  –  exact moments =======================================
library(moments)   # skewness & kurtosis

make_table2 <- function(df, set_name) {
  tibble(
    Metric = c("Customers (N)", "Zero-incidence rate", "Mean weekly spend (£)",
               "SD weekly spend (£)", "Skewness spend", "Kurtosis spend"),
    Value = c(
      n_distinct(df$customer_id),
      round(mean(df$WeeklySpend == 0), 2),
      round(mean(df$WeeklySpend), 2),
      round(sd(df$WeeklySpend), 2),
      round(skewness(df$WeeklySpend), 2),
      round(kurtosis(df$WeeklySpend), 2)
    ),
    Dataset = set_name
  )
}

system.time({ table2 <- bind_rows(make_table2(uci, "UCI"), make_table2(cdnow, "CDNOW")) }) # 0.0004 s
as.data.frame(table2)




# ===== 2.  point to your local files ========================================
#path_uci <- "https://archive.ics.uci.edu/static/public/352/data.csv"
#path_cdnow <- "/Users/sudhirvoleti/research related/HMM n tweedie in RFM Nov 2025/CDNOW/cdnow_raw_10pct.csv"               # <-- drop your cleaned CDNOW csv here

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
