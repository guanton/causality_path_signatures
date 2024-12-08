#!/bin/bash

# Assumes you have four config JSON files corresponding to each experiment.
# For demonstration, we inline the same configs you provided:
# You'd create these config files: simple_ou_config.json, pure_bm_config.json, etc.

# Run the first experiment (Ornstein-Uhlenbeck type)
python data_generation.py --config simple_ou_config.json --experiment_name simple_OU_1

# Run the second experiment (Pure Brownian motion)
python data_generation.py --config pure_bm_config.json --experiment_name pure_brownian_motion_1

# Run the third experiment (proper_level_2_drift_1)
python data_generation.py --config proper_level_2_drift_1_config.json --experiment_name proper_level_2_drift_1

# Run the fourth experiment (proper_level_2_drift_and_diffusion_1)
python data_generation.py --config proper_level_2_drift_and_diffusion_1_config.json --experiment_name proper_level_2_drift_and_diffusion_1
