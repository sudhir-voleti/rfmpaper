#!/usr/bin/env Rscript
# poisson_nb_baseline_n500_simple.R - Guaranteed working version

library(tidyverse)
library(MASS)

data_dir <- "/Users/sudhirvoleti/research related/HMM n tweedie in RFM Nov 2025/Jan_SMC_Runs/data"
results_dir <- "/Users/sudhirvoleti/research related/HMM n tweedie in RFM Nov 2025/Jan_SMC_Runs/results"

run_baselines <- function() {
  all_results <- list()
  
  for(dataset in c("uci", "cdnow")) {
    cat(sprintf("\n=== Processing %s ===\n", dataset))
    
    # Load
    df <- read_csv(file.path(data_dir, paste0(dataset, "_500.csv")), show_col_types = FALSE)
    cat(sprintf("Loaded: %d customers, %d rows, %.1f%% zeros\n", 
                length(unique(df$customer_id)), nrow(df), 100*mean(df$spend==0)))
    
    # Prep
    df <- df %>% mutate(
      trans = as.integer(spend > 0),
      log_recency = log(R_weeks + 1),
      log_frequency = log(F_run + 1),
      log_monetary = log(M_run + 1)
    )
    
    # Poisson
    cat("  Fitting Poisson...\n")
    m_pois <- glm(trans ~ log_recency + log_frequency + log_monetary, 
                  data = df, family = poisson)
    
    all_results[[paste0(dataset, "_poisson")]] <- tibble(
      dataset = dataset,
      model = "Poisson_GLM",
      N = length(unique(df$customer_id)),
      n_obs = nrow(df),
      log_likelihood = as.numeric(logLik(m_pois)),
      AIC = AIC(m_pois),
      BIC = BIC(m_pois),
      converged = TRUE
    )
    
    # NBD
    cat("  Fitting NBD...\n")
    m_nbd <- tryCatch({
      glm.nb(trans ~ log_recency + log_frequency + log_monetary, data = df)
    }, error = function(e) {
      cat(sprintf("    NBD failed: %s\n", conditionMessage(e)))
      NULL
    })
    
    if(!is.null(m_nbd)) {
      all_results[[paste0(dataset, "_nbd")]] <- tibble(
        dataset = dataset,
        model = "NBD_GLM",
        N = length(unique(df$customer_id)),
        n_obs = nrow(df),
        log_likelihood = as.numeric(logLik(m_nbd)),
        AIC = AIC(m_nbd),
        BIC = BIC(m_nbd),
        theta = m_nbd$theta,
        converged = TRUE
      )
    } else {
      all_results[[paste0(dataset, "_nbd")]] <- tibble(
        dataset = dataset,
        model = "NBD_GLM",
        N = length(unique(df$customer_id)),
        n_obs = nrow(df),
        log_likelihood = NA,
        AIC = NA,
        BIC = NA,
        theta = NA,
        converged = FALSE
      )
    }
  }
  
  # Combine and save
  results_df <- bind_rows(all_results)
  print(results_df)
  
  write_csv(results_df, file.path(results_dir, "baseline_poisson_nbd_N500.csv"))
  cat(sprintf("\nSaved to: %s\n", file.path(results_dir, "baseline_poisson_nbd_N500.csv")))
  
  return(results_df)
}

results <- run_baselines()
