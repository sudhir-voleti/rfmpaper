library(tidyverse)
library(ggraph)
library(igraph)
library(ggforce)

plot_hmm_transitions <- function(Gamma, 
                                 title = "Posterior Mean Transition Probabilities (K = 3)",
                                 subtitle = "10,000 MCMC draws", 
                                 digits = 3) {
  
  if (!all.equal(dim(Gamma), c(3,3))) stop("Gamma must be a 3×3 matrix")
  if (is.null(rownames(Gamma))) stop("Gamma must have row names (state names)")
  
  # Edge data
  edges <- as.data.frame(as.table(Gamma)) %>%
    set_names(c("from", "to", "prob")) %>%
    mutate(prob_label = sprintf(paste0("%.", digits, "f"), prob))
  
  self_loops  <- edges %>% filter(from == to)
  inter_edges <- edges %>% filter(from != to)
  
  # Graph (only inter-state edges for arrows)
  g <- graph_from_data_frame(inter_edges,
                             vertices = data.frame(name = rownames(Gamma)))
  
  # Fixed triangle layout — order matches rownames
  manual_layout <- data.frame(
    name = rownames(Gamma),
    x = c(0, 1.5, 3),
    y = c(0, 2, 0)
  )
  
  # Plot
  ggraph(g, layout = manual_layout) +
    # Curved inter-state arrows
    geom_edge_arc(aes(width = prob),
                  strength = 0.35,
                  arrow = arrow(length = unit(5, "mm"), type = "closed"),
                  start_cap = circle(18, "mm"),
                  end_cap = circle(18, "mm"),
                  colour = "grey30") +
    
    # White halo for edge labels
    geom_edge_arc(aes(label = prob_label),
                  strength = 0.35,
                  label_colour = "white",
                  label_size = 9,                   # thick halo
                  label_dodge = unit(6, "mm"),
                  angle_calc = "along",
                  force_flip = TRUE,
                  show.legend = FALSE) +
    
    # Black text on top
    geom_edge_arc(aes(label = prob_label),
                  strength = 0.35,
                  label_colour = "black",
                  label_size = 4.5,
                  label_dodge = unit(6, "mm"),
                  angle_calc = "along",
                  force_flip = TRUE,
                  show.legend = FALSE) +
    
    scale_edge_width(range = c(1.5, 7), guide = "none") +
    
    # Self-loops: dashed ovals (angle now inside aes!)
    geom_ellipse(data = self_loops,
                 aes(x0 = manual_layout$x[match(from, manual_layout$name)],
                     y0 = manual_layout$y[match(from, manual_layout$name)] + 0.4,
                     a = prob * 1.2,      # horizontal radius
                     b = 0.4,             # vertical radius
                     angle = 0),          # <-- must be in aes()
                 colour = "grey50", fill = NA, linetype = "dashed", linewidth = 1.2) +
    
    # White rectangular labels for self-loops
    geom_label(data = self_loops,
               aes(x = manual_layout$x[match(from, manual_layout$name)],
                   y = manual_layout$y[match(from, manual_layout$name)] + 0.9,
                   label = prob_label),
               fill = "white", colour = "black",
               size = 4.5, fontface = "bold", label.size = 0.5) +
    
    # Nodes
    geom_node_point(size = 35, fill = "white", shape = 21, stroke = 2, colour = "grey20") +
    geom_node_text(aes(label = name), fontface = "bold", size = 6) +
    
    coord_fixed(xlim = c(-0.6, 3.6), ylim = c(-0.6, 2.6)) +
    theme_void() +
    labs(title = title,
         subtitle = paste0(subtitle, "   •   Self-loop probabilities shown as dashed ovals")) +
    theme(plot.title    = element_text(hjust = 0.5, size = 14, face = "bold"),
          plot.subtitle = element_text(hjust = 0.5, size = 11, colour = "grey40"))
}

# ——————————————————
# Example usage
# ——————————————————

Gamma_uci <- matrix(c(0.318, 0.323, 0.359,
                      0.291, 0.290, 0.418,
                      0.270, 0.274, 0.456), nrow = 3, byrow = TRUE)
rownames(Gamma_uci) <- colnames(Gamma_uci) <- c("Engaged", "Cooling", "Churned")

p <- plot_hmm_transitions(Gamma_uci,
                          title = "UCI: Posterior Mean Transition Probabilities (K = 3)")

print(p)  # displays in R

ggsave("uci_hmm_final.pdf", plot = p, width = 10, height = 7, dpi = 400)

### ---- for CDNOW

Gamma_cdn <- matrix(c(0.456, 0.271, 0.273,
                      0.360, 0.322, 0.318,
                      0.418, 0.289, 0.293), nrow = 3, byrow = TRUE)
rownames(Gamma_cdn) <- colnames(Gamma_cdn) <- c("Engaged", "Cooling", "Churned")

plot_hmm_transitions(Gamma_cdn, title = "CDNOW: Posterior Mean Transition Probabilities (K = 3)")

ggsave("cdnow_hmm_final.pdf", plot = p, width = 10, height = 7, dpi = 400)

