#!/usr/bin/env Rscript
# poisson_nb_baseline_n500.R
# Run Poisson and NBD on N=500 samples for fair comparison with SMC results

library(tidyverse)
library(broom)
library(MASS)  # for glm.nb

# Config
data_dir <- "/Users/sudhirvoleti/research related/HMM n tweedie in RFM Nov 2025/Jan_SMC_Runs/data"
results_dir <- "/Users/sudhirvoleti/research related/HMM n tweedie in RFM Nov 2025/Jan_SMC_Runs/results"

# Load pre-extracted N=500 data
load_panel_data <- function(dataset, n = 500) {
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

# Prepare RFM features
prep_features <- function(df) {
  df %>%
    mutate(
      log_recency = log(R_weeks + 1),
      log_frequency = log(F_run + 1),
      log_monetary = log(M_run + 1)
    )
}

# Model 1: Poisson GLM
fit_poisson <- function(df, dataset) {
  # Fit Poisson on count of transactions (spend > 0)
  model <- glm((spend > 0) ~ log_recency + log_frequency + log_monetary, 
               data = df, family = poisson(link = "log"))
  
  loglik <- as.numeric(logLik(model))
  
  tibble(
    dataset = dataset,
    model = "Poisson_GLM",
    N = length(unique(df$customer_id)),
    log_likelihood = loglik,
    AIC = AIC(model),
    BIC = BIC(model),
    converged = TRUE
  )
}

# Model 2: Negative Binomial
fit_nbd <- function(df, dataset) {
  tryCatch({
    model <- glm.nb((spend > 0) ~ log_recency + log_frequency + log_monetary, 
                    data = df, control = glm.control(maxit = 100))
    
    tibble(
      dataset = dataset,
      model = "NBD_GLM",
      N = length(unique(df$customer_id)),
      log_likelihood = as.numeric(logLik(model)),
      AIC = AIC(model),
      BIC = BIC(model),
      theta = model$theta,
      converged = TRUE
    )
  }, error = function(e) {
    tibble(
      dataset = dataset,
      model = "NBD_GLM",
      N = length(unique(df$customer_id)),
      log_likelihood = NA,
      AIC = NA,
      BIC = NA,
      theta = NA,
      converged = FALSE,
      error = as.character(e)
    )
  })
}

# Main execution
run_baselines <- function() {
  all_results <- list()
  
  for(dataset in c("uci", "cdnow")) {
    cat(sprintf("\nProcessing %s...\n", dataset))
    
    df <- load_panel_data(dataset)
    df <- prep_features(df)
    
    cat("Fitting Poisson...\n")
    all_results[[paste0(dataset, "_poisson")]] <- fit_poisson(df, dataset)
    
    cat("Fitting NBD...\n")
    all_results[[paste0(dataset, "_nbd")]] <- fit_nbd(df, dataset)
  }
  
  results_df <- bind_rows(all_results)
  print(results_df)
  
  output_file <- file.path(results_dir, "baseline_poisson_nbd_N500.csv")
  write_csv(results_df, output_file)
  cat(sprintf("\nSaved to: %s\n", output_file))
  
  return(results_df)
}

# Run
results <- run_baselines()
