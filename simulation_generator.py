#!/usr/bin/env python3
"""
simulation_generator.py
=======================
Generate synthetic RFM panel with principled 3-state DGP + Dead state.
Designed to test HMM-Tweedie recovery of heterogeneous regimes.
"""

======================================================================
SMC Unified: SIMULATION | HURDLE | K=1
  Loaded: N=500, T_train=80, T_test=20, zeros=77.5% (from csv)
======================================================================

Loaded: N=500, T=80, zeros=77.5%

 Model: K=1, HURDLE-GLM
Initializing SMC sampler...
Sampling 4 chains in 4 jobs
PyTensor config: floatX=float32, optimizer=fast_run
Detected Apple Silicon (arm64). Float32 optimization enabled.
Chain 0 ⠼ -:--:-- / 0:00:00 Stage: 0 Beta: 0
Chain 1 ⠼ -:--:-- / 0:00:00 Stage: 0 Beta: 0
Chain 2 ⠼ -:--:-- / 0:00:00 Stage: 0 Beta: 0
Chain 3 ⠼ -:--:-- / 0:00:00 Stage: 0 Beta: 0PyTensor config: floatX=float32, optimizer=fast_run
Chain 0 ⠋ -:--:-- / 0:00:01 Stage: 0 Beta: 0
Chain 1 ⠋ -:--:-- / 0:00:01 Stage: 0 Beta: 0
Chain 2 ⠋ -:--:-- / 0:00:01 Stage: 0 Beta: 0
Chain 3 ⠋ -:--:-- / 0:00:01 Stage: 0 Beta: 0Detected Apple Silicon (arm64). Float32 optimization enabled.
Chain 0 ⠼ -:--:-- / 0:20:01 Stage: 270 Beta: 1.000
Chain 1 ⠼ -:--:-- / 0:20:01 Stage: 269 Beta: 1.000
Chain 2 ⠼ -:--:-- / 0:20:01 Stage: 271 Beta: 1.000
Chain 3 ⠼ -:--:-- / 0:20:01 Stage: 271 Beta: 1.000
  DEBUG: y_test mean=17.89, y_pred mean=12.08
  DEBUG: y_pred zeros=0.0%
  DEBUG: scale ratio = 1.4807
  OOS RMSE: 77.3996, MAE: 24.8579
 ✓ log_ev=-43187.11, time=20.1min

