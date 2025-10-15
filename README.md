# MacroAgingNature
Data and code for article: "A GDP Approach to Measure Returns on Investment in Aging Biology"

# Instructions to reproduce results
0. Install the Anaconda distribution of Python (https://www.anaconda.com/download).
1. Clone the repository.
2. Create a conda environment with the required packages:
   ```bash
   conda env create -f environment.yml
   ```
3. Run the bash script to reproduce the results. This computation will take a long time, potentially more than 24 hours. The first script `run_macro_aging.sh` runs the baseline scenario plus the nine main reform scenarios in the paper. The `run_macro_aging_sensitivity.sh` script runs a high and low sensitivity simulation on each of the 9 scenarios (18 simulations) plus it re-runs the baseline. So the following two scripts represent 29 runs of the macroeconomic model, each of which can take around an hour.
   ```bash
   bash run_macro_aging.sh
   bash run_macro_aging_sensitivity.sh
    ```
4. All results will be saved in the `ResultsInPaper/macro_aging_results.csv` file and `ResultsInPaper/macro_aging_sensitivity_results.csv`. Individual figures from the paper will be saved in the `ResultsInPaper/figures/` directory.
