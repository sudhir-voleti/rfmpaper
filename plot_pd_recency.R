library(tidyverse)

plot_pd_recency <- function(csv_path,
                            title = "Partial Dependence of Log-Recency on Expected Weekly Spend",
                            subtitle = "95% CrI bands from 1 000 posterior draws",
                            out_pdf = NULL) {
  
  d <- read_csv(csv_path)
  
  p <- ggplot(d, aes(x = log_recency, y = mean)) +
    geom_ribbon(aes(ymin = cri_lower, ymax = cri_upper),
                alpha = 0.25, fill = "steelblue") +
    geom_line(colour = "steelblue", linewidth = 1) +
    facet_wrap(~state, ncol = 3, labeller = label_value) +
    labs(x = "Log(Recency + 1)",
         y = "Expected Weekly Spend (£)",
         title = title,
         subtitle = subtitle) +
    theme_minimal(base_size = 14) +
    theme(strip.text = element_text(face = "bold"),
          plot.title = element_text(hjust = 0.5),
          plot.subtitle = element_text(hjust = 0.5, size = 11))
  
  if (!is.null(out_pdf)) ggsave(out_pdf, plot = p, width = 10, height = 3.5, dpi = 400)
  return(p)
}

setwd("/Users/sudhirvoleti/results/")

## ---- for UCI ----
p_uci <- plot_pd_recency("uci_pd_recency_1k.csv",
                         title = "UCI: State-Specific Recency Effects (K = 3)",
                         out_pdf = "uci_pd_recency_1k.pdf")
print(p_uci)

## --- for CDNOW ----
plot_pd_recency("cdnow_pd_recency_1k.csv",
                title = "CDNOW: State-Specific Recency Effects (K = 3)",
                subtitle = "95% CrI bands from 1 000 posterior draws",
                out_pdf = "cdnow_pd_recency_1k.pdf")
