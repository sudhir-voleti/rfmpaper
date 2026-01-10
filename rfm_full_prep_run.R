#!/usr/bin/env Rscript
# rfm_data_prep.R  ----------------------------------------------------------
#  * ingest raw CSV  (UCI / CDNOW)
#  * build full RFM panel
#  * export .rds ready for modelling
#  * optional diagnostics plots (interactive only)
# ---------------------------------------------------------------------------

# ---- 0.  auto-install any missing packages ---------------------------------
req_pkg <- c("tidyverse", "lubridate", "readxl", "curl", "janitor",
             "ggplot2", "scales", "dplyr", "tidyr", "ggrepel",
             "ggraph", "tidygraph", "openssl")   # openssl only for SHA if you keep it

for (p in req_pkg)
  if (!requireNamespace(p, quietly = TRUE))
    install.packages(p, repos = "https://cloud.r-project.org")
suppressPackageStartupMessages(library(tidyverse))


# ---- 1.  UCI online retail -------------------------------------------------
ingest_uci <- function() {
  tmp <- tempfile(fileext = ".csv")
  curl::curl_download("https://archive.ics.uci.edu/static/public/352/data.csv", tmp)

  raw <- read_csv(tmp, show_col_types = FALSE) %>%   # CSV, not Excel
    janitor::clean_names()

  unlink(tmp)

  raw %>%
    mutate(
      date        = as.Date(lubridate::mdy_hm(invoice_date)),  # US format
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

# ---- 2.  CDNOW 1/10 sample -------------------------------------------------
ingest_cdnow <- function() {
  tmp_dir <- tempdir()
  zip_path <- file.path(tmp_dir, "cdnow.zip")
  curl::curl_download("http://www.brucehardie.com/datasets/CDNOW_sample.zip", zip_path)
  unzip(zip_path, exdir = tmp_dir)
  txt_path <- list.files(tmp_dir, pattern = "\\.txt$", full.names = TRUE)[1]  # sample inside
  raw <- read_delim(txt_path, delim = " ", col_names = c("customer", "date", "dollar"), trim_ws = TRUE)
  
  raw %>%
    transmute(
      customer_id = as.character(customer),
      date        = as.Date(date, format = "%Y%m%d"),
      Monetary    = dollar,
      WeekStart   = floor_date(date, "week")
    ) %>%
    group_by(customer_id, WeekStart) %>%
    summarise(WeeklySpend = sum(Monetary), n_transactions = n(),
              .groups = "drop")
  return(raw)
}


# ---- 3.  end-to-end wrappers ---------------------------------------------
build_rfm_uci  <- function() build_rfm_baseline(ingest_uci())
build_rfm_cdnow<- function() build_rfm_baseline(ingest_cdnow())

# ---- 4.  batch build -------------------------------------------------------
if (sys.nframe() == 0) {               # Rscript
  cat("Fetching & building UCI panel...\n")
  uci_rfm  <- build_rfm_uci()
  saveRDS(uci_rfm,  "data/uci_rfm.rds")
  cat("UCI final panel:"); print(glimpse(uci_rfm))
  
  cat("Fetching & building CDNOW panel...\n")
  cdnow_rfm<- build_rfm_cdnow()
  saveRDS(cdnow_rfm, "data/cdnow_rfm.rds")
  cat("CDNOW final panel:"); print(glimpse(cdnow_rfm))
  
  cat("Panels cached in data/*.rds\n")
}

# ---- 2.  RFM baseline builder ---------------------------------------------
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
    mutate(
      R_weeks            = as.numeric(max_w - WeekStart) / 7,
      F_run              = cumsum(n_transactions),
      M_run              = cummean(WeeklySpend),
      zero_incidence_run = cumsum(WeeklySpend == 0) / row_number()
    ) %>%
    ungroup() %>%
    arrange(customer_id, WeekStart)
}

# ---- 3.  plotting helpers (from GitHub files) -----------------------------
# 3a. empirical spend surface  (ex plot_pd_recency.R)
plot_spend_surface <- function(rfm, set_name) {
  summ <- rfm %>%
    filter(WeeklySpend > 0) %>%               # positive only
    mutate(logSpend = log(WeeklySpend)) %>%
    group_by(R_weeks = round(R_weeks), zero_incidence_run = round(zero_incidence_run, 2)) %>%
    summarise(meanLogSpend = mean(logSpend, na.rm = TRUE), .groups = "drop")

  ggplot(summ, aes(R_weeks, zero_incidence_run, fill = meanLogSpend)) +
    geom_tile() +
    scale_fill_viridis_c(option = "magma", na.value = "grey90") +
    labs(title = paste(set_name, "– empirical spend surface"),
         x = "Recency (weeks)", y = "Zero-incidence rate", fill = "log(spend)") +
    theme_minimal()
}

# 3b. transition-matrix network  (ex transMat_plot.R)
plot_trans_mat <- function(gamma_mat, set_name, K = 3) {
  gamm_df <- gamma_mat %>%                  # gamma_mat = posterior mean matrix (K x K)
    as.data.frame() %>% mutate(from = row_number()) %>%
    pivot_longer(-from, names_to = "to", values_to = "prob") %>%
    mutate(to   = as.numeric(to),
           from = factor(from, levels = 1:K, labels = c("Hot", "Warm", "Cold")),
           to   = factor(to,   levels = 1:K, labels = c("Hot", "Warm", "Cold")))

  as_tbl_graph(gamm_df, directed = TRUE) %>%
    activate(edges) %>% filter(prob > 0.02) %>%        # drop near-zero arrows
    ggraph(layout = "fr") +
    geom_edge_link(aes(width = prob), arrow = arrow(length = unit(2, "mm")), colour = "steelblue") +
    geom_node_text(aes(label = name), size = 5, colour = "black") +
    scale_edge_width(range = c(0.2, 2)) +
    labs(title = paste(set_name, "– latent transition network")) +
    theme_void()
}

# ---- 4.  end-to-end wrappers ---------------------------------------------
build_rfm_uci  <- function(path_raw) build_rfm_baseline(ingest_uci(path_raw))
build_rfm_cdnow<- function(path_raw) build_rfm_baseline(ingest_cdnow(path_raw))

# ---- 5.  batch build (silent) --------------------------------------------
if (sys.nframe() == 0) {                 # executed via Rscript
  here::i_am("rfm_data_prep.R")
  cat("Building UCI panel...\n")
  uci_rfm  <- build_rfm_uci (here("data", "Online Retail.csv"))
  saveRDS(uci_rfm,  here("data", "uci_rfm.rds"))

  cat("Building CDNOW panel...\n")
  cdnow_rfm<- build_rfm_cdnow(here("data", "cdnow.csv"))
  saveRDS(cdnow_rfm, here("data", "cdnow_rfm.rds"))

  cat("Panels saved to data/*.rds\n")
}

# ---- 6.  interactive diagnostics (only when sourced in RStudio) -----------
if (interactive()) {
  print("Generating diagnostic plots...")
  plot_spend_surface(uci_rfm,  "UCI")
  plot_spend_surface(cdnow_rfm,"CDNOW")

  # (gamma_mat needs your HMM posterior – stub now, fill after SMC)
  # gamma_uci <- matrix(c(...), nrow = 3, ncol = 3)   # posterior mean
  # plot_trans_mat(gamma_uci, "UCI")
}
