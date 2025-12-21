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
library(ggraph)    # For HMM Network plots
library(igraph)    # For Graph structures
library(ggforce)   # For Ellipse geoms

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

# --- 5. MANUSCRIPT TABLE BUILDERS ---

#' Table 1: Comparative Summary Statistics
make_manuscript_table_1 <- function(rfm_uci, rfm_cdnow) {
  calc_stats <- function(df, label) {
    df %>%
      summarise(
        Dataset = label,
        `Total Obs (N)` = n(),
        `Unique Cust`   = n_distinct(customer_id),
        `Mean Weekly $` = round(mean(WeeklySpend), 2),
        `Zero-Inflation %` = paste0(round(mean(WeeklySpend == 0) * 100, 1), "%"),
        `Spend Skewness` = round(moments::skewness(WeeklySpend), 2),
        `Spend Kurtosis` = round(moments::kurtosis(WeeklySpend), 2)
      )
  }
  bind_rows(calc_stats(rfm_uci, "UCI Retail"), calc_stats(rfm_cdnow, "CDNOW"))
}

#' Table 2: Threshold Tipping Point Evidence
make_manuscript_table_2 <- function(rfm_df) {
  cust_stats <- rfm_df %>%
    group_by(customer_id) %>%
    summarise(p0 = mean(WeeklySpend == 0), .groups = "drop")
  
  rfm_df %>%
    left_join(cust_stats, by = "customer_id") %>%
    mutate(Regime = case_when(
      p0 < 0.50  ~ "Low Zero (Active)",
      p0 < 0.75  ~ "Mid Zero (Intermittent)",
      TRUE       ~ "High Zero (Clumpy/Tipping)"
    )) %>%
    group_by(Regime) %>%
    summarise(
      `N Weeks` = n(),
      `Avg Spend` = round(mean(WeeklySpend), 2),
      `Pr(Y=0)` = round(mean(WeeklySpend == 0), 3),
      `Avg Freq` = round(mean(F_rolling), 2)
    )
}

# ==============================================================================
# USAGE EXAMPLES: REPLICATING MANUSCRIPT TABLES
# ==============================================================================
#
# # 1. DATA INGESTION & PREP
# uci_raw   <- ingest_uci("Online Retail.csv")
# cdn_raw   <- ingest_cdnow("purchases.csv")
#
# uci_panel <- build_rfm_baseline(uci_raw)
# cdn_panel <- build_rfm_baseline(cdn_raw)
#
# # 2. REPLICATE TABLE 1 (Summary Statistics)
# # Compares raw geometry and clumpiness (skewness/kurtosis)
# table_1 <- make_manuscript_table_1(uci_panel, cdn_panel)
# print(table_1)
#
# # 3. REPLICATE TABLE 2 (Tipping Point Evidence)
# # Tests the pi_0 approx 0.75 threshold
# table_2_cdn <- make_manuscript_table_2(cdn_panel)
# print(table_2_cdn)
#
# # 4. REPLICATE TABLE 3 (Static Horse-Race / Ablation Study)
# # Compares Poisson GLM vs. Neg-Binomial vs. Tweedie GAM
# table_3_uci <- run_static_benchmarks(uci_panel)
# table_3_cdn <- run_static_benchmarks(cdn_panel)
#
# print("UCI Results:")
# print(table_3_uci)
# print("CDNOW Results:")
# print(table_3_cdn)
#
# ==============================================================================

library(ggplot2)
library(patchwork) # For combining plots

# --- 6. MANUSCRIPT VISUALIZATION MODULE ---

#' Figure A: The Stochastic Gap (Distributional Comparison)
#' Visualizes actual spend vs. Model predictions to show the "Zero-Spike" fit
plot_stochastic_gap <- function(rfm_df, tweedie_mod) {
  preds <- predict(tweedie_mod, rfm_df, type = "response")
  
  plot_data <- data.frame(
    Value = c(rfm_df$WeeklySpend, preds),
    Type = rep(c("Actual Spend", "Tweedie Prediction"), each = nrow(rfm_df))
  )
  
  ggplot(plot_data, aes(x = Value, fill = Type)) +
    geom_histogram(bins = 50, alpha = 0.6, position = "identity") +
    scale_x_log10(labels = scales::dollar_format()) +
    theme_minimal() +
    labs(title = "The Stochastic Gap: Spend Distribution vs. Tweedie Fit",
         subtitle = "Visualizing zero-inflation and heavy-tail capture",
         x = "Weekly Spend (Log Scale)", y = "Frequency Count") +
    scale_fill_manual(values = c("grey30", "firebrick"))
}

#' Figure B: The Recency Cliff (Non-linear Marginal Effects)
#' Replicates the visual proof of structural collapse in traditional models
plot_recency_cliff <- function(tweedie_gam) {
  # Extract marginal effect of Recency from the GAM
  plot_df <- plot(tweedie_gam, select = 1, shift = coef(tweedie_gam)[1], 
                  trans = exp, seWithMean = TRUE, unconditional = TRUE)
  
  # We manually rebuild the plot for ggplot aesthetics
  r_grid <- seq(min(tweedie_gam$model$R_lagged), max(tweedie_gam$model$R_lagged), length.out = 100)
  pred_data <- data.frame(R_lagged = r_grid, F_rolling = mean(tweedie_gam$model$F_rolling), 
                          M_rolling = mean(tweedie_gam$model$M_rolling))
  
  p_vals <- predict(tweedie_gam, pred_data, type = "link", se.fit = TRUE)
  pred_data$fit <- exp(p_vals$fit)
  pred_data$up  <- exp(p_vals$fit + 1.96 * p_vals$se.fit)
  pred_data$lo  <- exp(p_vals$fit - 1.96 * p_vals$se.fit)
  
  ggplot(pred_data, aes(x = R_lagged, y = fit)) +
    geom_ribbon(aes(ymin = lo, ymax = up), alpha = 0.2, fill = "royalblue") +
    geom_line(color = "royalblue", size = 1) +
    theme_minimal() +
    labs(title = "The Recency Cliff",
         subtitle = "Non-linear decay of expected spend as a function of Recency",
         x = "Days Since Last Purchase (R_lagged)", y = "Expected Weekly Spend")
}

#' Figure C: Regime Variance Diagnostic
#' Replicates the 'Threshold Tipping Point' visual evidence
plot_regime_variance <- function(rfm_df) {
  # Create regimes based on our manuscript threshold (pi_0 = 0.75)
  cust_p0 <- rfm_df %>% group_by(customer_id) %>% summarise(p0 = mean(WeeklySpend == 0))
  
  rfm_df %>%
    left_join(cust_p0, by = "customer_id") %>%
    mutate(Regime = if_else(p0 >= 0.75, "High-Zero (Tipping)", "Low-Zero (Active)")) %>%
    ggplot(aes(x = Regime, y = WeeklySpend + 1, fill = Regime)) +
    geom_violin(alpha = 0.7) +
    scale_y_log10() +
    theme_minimal() +
    labs(title = "Variance Explosion at the Tipping Point",
         y = "Spend + 1 (Log Scale)") +
    guides(fill = "none")
}

# # 5. REPLICATE MANUSCRIPT FIGURES
# # Fit the Tweedie GAM first (from our horse-race module)
# uci_gam <- gam(WeeklySpend ~ s(R_lagged) + s(F_rolling) + s(M_rolling), 
#                family = Tweedie(p = 1.25, link = "log"), data = uci_panel)
#
# # Replicate the Distributional Fit Figure
# fig_1 <- plot_stochastic_gap(uci_panel, uci_gam)
# ggsave("figure_1_stochastic_gap.png", fig_1, width = 8, height = 5)
#
# # Replicate the Recency Cliff Figure
# fig_2 <- plot_recency_cliff(uci_gam)
# ggsave("figure_2_recency_cliff.png", fig_2, width = 8, height = 5)
#
# # Replicate the Tipping Point Diagnostic
# fig_3 <- plot_regime_variance(cdn_panel)
# ggsave("figure_3_regime_variance.png", fig_3, width = 8, height = 5)

#' Figure D: HMM Transition Network (K=3)
#' Visualizes the latent state dynamics of the 'Boiling Pot' model
plot_hmm_transitions <- function(Gamma, 
                                 title = "Posterior Mean Transition Probabilities (K = 3)",
                                 subtitle = "10,000 MCMC draws", 
                                 digits = 3) {
  
  if (!all.equal(dim(Gamma), c(3,3))) stop("Gamma must be a 3x3 matrix")
  if (is.null(rownames(Gamma))) stop("Gamma must have row names (state names)")
  
  # Edge data
  edges <- as.data.frame(as.table(Gamma)) %>%
    set_names(c("from", "to", "prob")) %>%
    mutate(prob_label = sprintf(paste0("%.", digits, "f"), prob))
  
  self_loops  <- edges %>% filter(from == to)
  inter_edges <- edges %>% filter(from != to)
  
  # Graph structure
  g <- graph_from_data_frame(inter_edges, vertices = data.frame(name = rownames(Gamma)))
  
  # Fixed triangle layout
  manual_layout <- data.frame(
    name = rownames(Gamma),
    x = c(0, 1.5, 3),
    y = c(0, 2, 0)
  )
  
  # Plotting
  ggraph(g, layout = manual_layout) +
    geom_edge_arc(aes(width = prob), strength = 0.35, colour = "grey30",
                  arrow = arrow(length = unit(5, "mm"), type = "closed"),
                  start_cap = circle(18, "mm"), end_cap = circle(18, "mm")) +
    # White halo for labels
    geom_edge_arc(aes(label = prob_label), strength = 0.35, label_colour = "white",
                  label_size = 9, label_dodge = unit(6, "mm"), angle_calc = "along") +
    # Black text
    geom_edge_arc(aes(label = prob_label), strength = 0.35, label_colour = "black",
                  label_size = 4.5, label_dodge = unit(6, "mm"), angle_calc = "along") +
    scale_edge_width(range = c(1.5, 7), guide = "none") +
    # Self-loops as ovals
    geom_ellipse(data = self_loops,
                 aes(x0 = manual_layout$x[match(from, manual_layout$name)],
                     y0 = manual_layout$y[match(from, manual_layout$name)] + 0.4,
                     a = prob * 1.2, b = 0.4, angle = 0),
                 colour = "grey50", fill = NA, linetype = "dashed", linewidth = 1.2) +
    geom_label(data = self_loops,
               aes(x = manual_layout$x[match(from, manual_layout$name)],
                   y = manual_layout$y[match(from, manual_layout$name)] + 0.9,
                   label = prob_label),
               fill = "white", fontface = "bold") +
    geom_node_point(size = 35, fill = "white", shape = 21, stroke = 2, colour = "grey20") +
    geom_node_text(aes(label = name), fontface = "bold", size = 6) +
    coord_fixed(xlim = c(-0.6, 3.6), ylim = c(-0.6, 2.6)) +
    theme_void() +
    labs(title = title, subtitle = paste0(subtitle, " • Self-loop probabilities shown as dashed ovals")) +
    theme(plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
          plot.subtitle = element_text(hjust = 0.5, size = 11, colour = "grey40"))
}

# ==============================================================================
# USAGE EXAMPLES
# ==============================================================================
# # 1. DATA PREP
# uci <- build_rfm_baseline(ingest_uci("Online Retail.csv"))
#
# # 2. TABLES
# make_manuscript_table_1(uci, uci) # Replace with CDNOW for full comparison
#
# # 3. VISUALIZATIONS (Static & HMM)
# # (A) Recency Cliff
# mod_gam <- gam(WeeklySpend ~ s(R_lagged) + s(F_rolling), family = Tweedie(p=1.25), data = uci)
# plot_recency_cliff(mod_gam)
#
# # (B) HMM Transitions (Using values from Python/SMC run)
# Gamma_uci <- matrix(c(0.318, 0.323, 0.359, 
#                       0.291, 0.290, 0.418, 
#                       0.270, 0.274, 0.456), nrow = 3, byrow = TRUE)
# rownames(Gamma_uci) <- colnames(Gamma_uci) <- c("Engaged", "Cooling", "Churned")
# plot_hmm_transitions(Gamma_uci, title = "UCI: Latent State Dynamics")
