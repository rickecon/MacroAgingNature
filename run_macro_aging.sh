#!/bin/bash

# Activate Conda environment
source activate macro-aging-env

# Run simulations
cd SimulationCode
# Main results
python simulation.py scenarios.json
# Sensitivity results
python simulation.py hi_low_parameterizations.json

# process results and save to CSV
cd ..
mkdir -p Results
cd SimulationCode
python process_results.py "simulation_results/" scenarios.json "../ResultsInPaper/macro_aging_results.csv"
python process_results.py "simulation_results/" hi_low_parameterizations.json "../ResultsInPaper/macro_aging_sensitivity_results.csv"
python figures.py
