library(tidyverse)
library(ggraph)
library(igraph)
library(ggforce)

plot_hmm_transitions_high_contrast <- function(Gamma, dataset_name = "Dataset") {
  require(igraph)
  K <- nrow(Gamma)
  
  # 1. Geometry
  angles <- seq(pi/2, pi/2 - 2*pi, length.out = K + 1)[1:K]
  pts <- cbind(cos(angles), sin(angles))
  
  # 2. Build Graph
  g <- graph_from_adjacency_matrix(Gamma, weighted = TRUE, diag = TRUE)
  
  # 3. Enhanced Aesthetics
  # We use a log-ish scaling or a floor to ensure small edges don't disappear
  # This makes the 0.001 edges visible but the 0.99 edges clearly dominant
  E(g)$width <- (E(g)$weight * 10) + 0.5 
  
  # Force a dark, consistent color for all visible edges
  edge_col <- adjustcolor("grey20", alpha.f = 0.8) 
  
  # Labels only for substantial transitions to keep it clean
  E(g)$label <- ifelse(E(g)$weight > 0.03, round(E(g)$weight, 2), "")
  
  # 4. The Plot
  par(mar = c(1, 1, 1, 1))
  plot(g, 
       layout = pts,
       edge.arrow.size = 0.5,       # Slightly larger arrows
       edge.arrow.width = 1.2,      # Braoder arrowheads for visibility
       edge.curved = 0.25,
       edge.color = edge_col,       # Consistent dark gray
       edge.label = E(g)$label,
       edge.label.cex = 0.8,
       edge.label.color = "black",
       edge.label.font = 2,         # Bold labels
       edge.loop.angle = angles,
       edge.loop.size = 1.6,        # Pushed slightly further out
       vertex.size = 28,
       vertex.color = "white",
       vertex.frame.color = "black",
       vertex.frame.width = 1.5,
       vertex.label.cex = 1.1,
       vertex.label.font = 2,
       vertex.label = paste0("S", 0:(K-1)),
       main = paste0("HMM Transitions: ", dataset_name, " (K=", K, ")"))
}


# ——————————————————
# Example usage
# ——————————————————
# UCI Transition Matrix (K=5)
# Rows: From State (0-4), Cols: To State (0-4)
gamma_uci <- matrix(c(
  0.160221, 0.065635, 0.260256, 0.401197, 0.112675, # From State 0
  0.054096, 0.131152, 0.079141, 0.599510, 0.136082, # From State 1
  0.100929, 0.196401, 0.348891, 0.215821, 0.137972, # From State 2
  0.200281, 0.095453, 0.310740, 0.366014, 0.027490, # From State 3
  0.096869, 0.391955, 0.365728, 0.021374, 0.124042  # From State 4
), nrow = 5, byrow = TRUE)

# Assign state names for the plot
colnames(gamma_uci) <- rownames(gamma_uci) <- c("S0", "S1", "S2", "S3", "S4")

plot_hmm_transitions_high_contrast(gamma_uci, "UCI Online Retail")



## ---- CDNOW Transition Matrix (K=4)
# Rows: From State (0-3), Cols: To State (0-3)
gamma_cdnow <- matrix(c(
  0.658361, 0.154916, 0.002700, 0.184017, # From State 0
  0.117717, 0.715590, 0.001322, 0.165366, # From State 1
  0.002635, 0.001810, 0.991690, 0.003865, # From State 2
  0.173029, 0.218621, 0.499345, 0.109006  # From State 3
), nrow = 4, byrow = TRUE)

# Assign state names
colnames(gamma_cdnow) <- rownames(gamma_cdnow) <- c("S0", "S1", "S2", "S3")

# Call your generic function
plot_hmm_transitions_high_contrast(gamma_cdnow, "CDNOW Marketplace")
