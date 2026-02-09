#!/usr/bin/env Rscript
# poisson_nb_baseline_n500.R
# Run Poisson and NBD on N=500 samples for fair comparison with SMC results

library(tidyverse)
library(broom)
library(MASS)  # for glm.nb
library(pscl)  # for zeroinfl if needed

# Config
data_dir <- "/Users/sudhirvoleti/research related/HMM n tweedie in RFM Nov 2025/Jan_SMC_Runs/data"
results_dir <- "/Users/sudhirvoleti/research related/HMM n tweedie in RFM Nov 2025/Jan_SMC_Runs/results"

# Load data function (matching your panel structure)
load_panel_data <- function(dataset, n = 500) {
  if(dataset == "uci") {
    df <- read_csv(file.path(data_dir, "uci_full.csv"))  # adjust filename if different
  } else if(dataset == "cdnow") {
    df <- read_csv(file.path(data_dir, "cdnow_full.csv"))  # adjust filename if different
  }
  
  # Sample n customers if needed
  customers <- unique(df$customer_id)
  if(length(customers) > n) {
    set.seed(42)  # for reproducibility
    sampled <- sample(customers, n)
    df <- df %>% filter(customer_id %in% sampled)
  }
  
  return(df)
}

# Prepare RFM features (matching your covariates)
prep_features <- function(df) {
  df %>%
    group_by(customer_id) %>%
    mutate(
      recency = as.numeric(difftime(max(week), week, units = "weeks")),
      frequency = cumsum(spend > 0),
      monetary = cummean(ifelse(spend > 0, spend, NA)) %>% zoo::na.locf(na.rm = FALSE) %>% replace_na(0)
    ) %>%
    ungroup() %>%
    mutate(
      log_recency = log(recency + 1),
      log_frequency = log(frequency + 1),
      log_monetary = log(monetary + 1)
    )
}

# Model 1: Poisson GLM (count of transactions)
fit_poisson <- function(df, dataset) {
  # Aggregate to customer-week level (count transactions per week)
  df_agg <- df %>%
    group_by(customer_id, week) %>%
    summarise(y = sum(spend > 0), .groups = "drop")  # count transactions
  
  # Fit Poisson on count
  model <- glm(y ~ log_recency + log_frequency + log_monetary, 
               data = df, family = poisson(link = "log"))
  
  # Log-likelihood
  loglik <- logLik(model)
  aic <- AIC(model)
  bic <- BIC(model)
  
  # For comparison with SMC log-evidence: use logLik (approximate)
  # Note: Poisson is poor fit for clumpy data, expect very negative loglik
  
  results <- tibble(
    dataset = dataset,
    model = "Poisson_GLM",
    N = length(unique(df$customer_id)),
    log_likelihood = as.numeric(loglik),
    AIC = aic,
    BIC = bic,
    deviance = deviance(model),
    df.residual = df.residual(model)
  )
  
  return(list(results = results, model = model))
}

# Model 2: Negative Binomial (NBD) - allows overdispersion
fit_nbd <- function(df, dataset) {
  # Same aggregation
  df_agg <- df %>%
    group_by(customer_id, week) %>%
    summarise(y = sum(spend > 0), .groups = "drop")
  
  # Try NBD - may fail to converge on clumpy data (as per your theory)
  tryCatch({
    model <- glm.nb(y ~ log_recency + log_frequency + log_monetary, 
                    data = df, control = glm.control(maxit = 100))
    
    loglik <- logLik(model)
    
    results <- tibble(
      dataset = dataset,
      model = "NBD_GLM", 
      N = length(unique(df$customer_id)),
      log_likelihood = as.numeric(loglik),
      AIC = AIC(model),
      BIC = BIC(model),
      theta = model$theta,  # dispersion parameter
      converged = TRUE
    )
    return(list(results = results, model = model))
    
  }, error = function(e) {
    # NBD often fails on extreme zero-inflation (>75%)
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
    
    # Load N=500 sample
    df_raw <- load_panel_data(dataset, n = 500)
    df <- prep_features(df_raw)
    
    # Poisson
    cat("Fitting Poisson...\n")
    pois_res <- fit_poisson(df, dataset)
    all_results[[paste0(dataset, "_poisson")]] <- pois_res$results
    
    # NBD
    cat("Fitting NBD...\n")
    nbd_res <- fit_nbd(df, dataset)
    if(is.list(nbd_res) && "results" %in% names(nbd_res)) {
      all_results[[paste0(dataset, "_nbd")]] <- nbd_res$results
    } else {
      all_results[[paste0(dataset, "_nbd")]] <- nbd_res
    }
  }
  
  # Combine and save
  results_df <- bind_rows(all_results)
  print(results_df)
  
  output_file <- file.path(results_dir, "baseline_poisson_nbd_N500.csv")
  write_csv(results_df, output_file)
  cat(sprintf("\nSaved to: %s\n", output_file))
  
  return(results_df)
}

# Run
results <- run_baselines()
