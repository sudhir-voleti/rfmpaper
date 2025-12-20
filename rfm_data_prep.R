#' RFM Methodological Toolkit: HMM-Tweedie-SMC Framework
#' Full Pipeline: Ingestion -> RFM Engine -> Diagnostics -> Static Benchmarks
#' Repository: sudhir-voleti/rfmpaper

library(tidyverse)
library(lubridate)
library(zoo)
library(mgcv)      # For Tweedie GAMs
library(MASS)      # For Negative Binomial
library(caret)     # For RMSE/MAE splitting
library(moments)   # For Skewness/Kurtosis

# --- 1. DATA INGESTION (Raw Files) ---

ingest_uci <- function(path) {
  read_csv(path) %>%
    mutate(
      date = dmy_hm(InvoiceDate), 
      Monetary = Quantity * UnitPrice,
      customer_id = as.character(CustomerID)
    ) %>%
    filter(!is.na(customer_id), Quantity > 0, UnitPrice > 0, !str_detect(InvoiceNo, "^C")) %>%
    group_by(customer_id, WeekStart = floor_date(date, "week")) %>%
    summarise(WeeklySpend = sum(Monetary), n_transactions = n_distinct(InvoiceNo), .groups = "drop")
}

ingest_cdnow <- function(path) {
  read.csv(path) %>%
    mutate(
      date = as_date(date),
      customer_id = as.character(users_id),
      WeekStart = floor_date(date, "week")
    ) %>%
    group_by(customer_id, WeekStart) %>%
    summarise(WeeklySpend = sum(amt), n_transactions = n(), .groups = "drop")
}

# --- 2. CORE RFM ENGINE (Lagged Feature Engineering) ---

build_rfm_baseline <- function(weekly_df) {
  weekly_df %>%
    group_by(customer_id) %>%
    complete(WeekStart = seq(min(WeekStart), max(WeekStart), by = "week"), 
             fill = list(WeeklySpend = 0, n_transactions = 0)) %>%
    arrange(WeekStart) %>%
    mutate(
      # Frequency: Lagged 4-week moving average
      F_rolling = lag(rollapplyr(n_transactions, width = 4, FUN = mean, fill = NA, partial = TRUE)),
      # Monetary: Lagged 4-week moving average
      M_rolling = lag(rollapplyr(WeeklySpend, width = 4, FUN = mean, fill = NA, partial = TRUE)),
      # Recency: Days since last activity (Known at start of week)
      was_active = lag(n_transactions > 0, default = FALSE),
      last_active = na.locf(if_else(was_active, lag(WeekStart), as.Date(NA)), na.rm = FALSE),
      R_lagged = as.numeric(difftime(WeekStart, coalesce(last_active, first(WeekStart)), units = "days"))
    ) %>%
    filter(!is.na(F_rolling), !is.na(M_rolling)) %>%
    ungroup()
}

# --- 3. DIAGNOSTICS & MOMENTS (The "Tipping Point" Analysis) ---

get_clumpy_diagnostics <- function(rfm_df, dataset_label = "Dataset") {
  rfm_df %>%
    group_by(customer_id) %>%
    summarise(p0_cust = mean(WeeklySpend == 0), .groups = "drop") %>%
    summarise(
      Dataset = dataset_label,
      N_Cust = n(),
      Avg_pi0 = mean(p0_cust),
      Max_pi0 = max(p0_cust),
      Skew_Spend = skewness(rfm_df$WeeklySpend),
      Zero_Pct = mean(rfm_df$WeeklySpend == 0)
    )
}

# --- 4. STATIC MODEL HORSE-RACE (Ablation Study) ---

run_static_benchmarks <- function(rfm_df) {
  # Chronological split: Train on first 80% of weeks, test on last 20%
  cut_date <- quantile(rfm_df$WeekStart, 0.8)
  train <- rfm_df %>% filter(WeekStart <= cut_date)
  test  <- rfm_df %>% filter(WeekStart > cut_date)
  
  # A. Poisson GLM
  mod_poisson <- glm(WeeklySpend ~ R_lagged + F_rolling + M_rolling, 
                     family = poisson(link = "log"), data = train)
  
  # B. Negative Binomial (CRM Standard)
  mod_nb <- tryCatch(
    glm.nb(WeeklySpend ~ R_lagged + F_rolling + M_rolling, data = train),
    error = function(e) return(NULL)
  )
  
  # C. Tweedie GAM (The "Proposed" Stochastic Fix)
  # Uses p=1.25 as identified in your paper for ZIG limits
  mod_tweedie <- gam(WeeklySpend ~ s(R_lagged) + s(F_rolling) + s(M_rolling), 
                     family = Tweedie(p = 1.25, link = "log"), data = train)
  
  # Helper for performance metrics
  calc_eval <- function(model, data, name) {
    if(is.null(model)) return(NULL)
    
    preds <- predict(model, data, type = "response")
    actuals <- data$WeeklySpend
    
    # Calculate MAE and RMSE
    mae <- mean(abs(actuals - preds))
    rmse <- sqrt(mean((actuals - preds)^2))
    
    # Pseudo R-squared (Deviance based)
    r2 <- 1 - (deviance(model) / model$null.deviance)
    
    data.frame(Model = name, MAE = mae, RMSE = rmse, R2 = r2)
  }
  
  # Combine results
  results <- list(
    calc_eval(mod_poisson, test, "Poisson GLM"),
    calc_eval(mod_nb, test, "Neg-Binomial"),
    calc_eval(mod_tweedie, test, "Tweedie GAM")
  )
  
  do.call(rbind, results)
}
