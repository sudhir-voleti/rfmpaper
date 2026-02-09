#!/usr/bin/env Rscript
# poisson_nb_baseline_n500.R - FIXED VERSION

library(tidyverse)
library(broom)
library(MASS)

# Config
data_dir <- "/Users/sudhirvoleti/research related/HMM n tweedie in RFM Nov 2025/Jan_SMC_Runs/data"
results_dir <- "/Users/sudhirvoleti/research related/HMM n tweedie in RFM Nov 2025/Jan_SMC_Runs/results"

# Load pre-extracted N=500 data
load_panel_data <- function(dataset) {
  file_path <- file.path(data_dir, paste0(dataset, "_500.csv"))
  
  if(!file.exists(file_path)) {
    stop(paste("File not found:", file_path, "- Run Python extraction first"))
  }
  
  df <- read_csv(file_path, show_col_types = FALSE)
  
  cat(sprintf("Loaded %s_500.csv: %d customers, %d observations, %.1f%% zeros\n", 
              dataset, 
              length(unique(df$customer_id)),
              nrow(df),
              100 * mean(df$spend == 0, na.rm = TRUE)))
  
  return(df)
}

# Prepare features - aggregate to customer-week level for count models
prep_features <- function(df) {
  df %>%
    # Create transaction indicator (1 if spend > 0, else 0)
    mutate(trans = as.integer(spend > 0)) %>%
    # Ensure R_weeks, F_run, M_run exist
    mutate(
      log_recency = log(R_weeks + 1),
      log_frequency = log(F_run + 1),
      log_monetary = log(M_run + 1)
    )
}

# Model 1: Poisson GLM
fit_poisson <- function(df, dataset) {
  cat("  Fitting Poisson...\n")
  
  # Use transaction count as response (already 0/1 per week per customer)
  model <- glm(trans ~ log_recency + log_frequency + log_monetary, 
               data = df, 
               family = poisson(link = "log"))
  
  if(!inherits(model, "glm")) {
    stop("Poisson model fitting failed")
  }
  
  loglik <- as.numeric(logLik(model))
  
  tibble(
    dataset = dataset,
    model = "Poisson_GLM",
    N = length(unique(df$customer_id)),
    log_likelihood = loglik,
    AIC = AIC(model),
    BIC = BIC(model),
    converged = model$converged,
    n_obs = nrow(df)
  )
}

# Model 2: Negative Binomial
fit_nbd <- function(df, dataset) {
  cat("  Fitting NBD...\n")
  
  tryCatch({
    model <- glm.nb(trans ~ log_recency + log_frequency + log_monetary, 
                    data = df, 
                    control = glm.control(maxit = 100))
    
    if(!inherits(model, "negbin")) {
      stop("NBD model fitting failed")
    }
    
    tibble(
      dataset = dataset,
      model = "NBD_GLM",
      N = length(unique(df$customer_id)),
      log_likelihood = as.numeric(logLik(model)),
      AIC = AIC(model),
      BIC = BIC(model),
      theta = model$theta,
      converged = TRUE,
      n_obs = nrow(df)
    )
  }, error = function(e) {
    cat(sprintf("  NBD failed: %s\n", conditionMessage(e)))
    tibble(
      dataset = dataset,
      model = "NBD_GLM",
      N = length(unique(df$customer_id)),
      log_likelihood = NA_real_,
      AIC = NA_real_,
      BIC = NA_real_,
      theta = NA_real_,
      converged = FALSE,
      error = conditionMessage(e),
      n_obs = nrow(df)
    )
  })
}

# Main execution
run_baselines <- function() {
  all_results <- list()
  
  for(dataset in c("uci", "cdnow")) {
    cat(sprintf("\n=== Processing %s ===\n", dataset))
    
    df <- load_panel_data(dataset)
    df <- prep_features(df)
    
    # Check for missing values in key columns
    if(any(is.na(df$log_recency)) || any(is.na(df$log_frequency)) || any(is.na(df$log_monetary))) {
      cat("Warning: Missing values in covariates, removing...\n")
      df <- df %>% filter(!is.na(log_recency), !is.na(log_frequency), !is.na(log_monetary))
    }
    
    cat(sprintf("Final rows for modeling: %d\n", nrow(df)))
    
    # Fit models
    all_results[[paste0(dataset, "_poisson")]] <- fit_poisson(df, dataset)
    all_results[[paste0(dataset, "_nbd")]] <- fit_nbd(df, dataset)
  }
  
  results_df <- bind_rows(all_results)
  
  cat("\n=== RESULTS ===\n")
  print(results_df)
  
  output_file <- file.path(results_dir, "baseline_poisson_nbd_N500.csv")
  write_csv(results_df, output_file)
  cat(sprintf("\nSaved to: %s\n", output_file))
  
  return(results_df)
}

# Run
results <- run_baselines()
