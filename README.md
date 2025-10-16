# MacroAgingNature
Data and code for article: "A GDP Approach to Measure Returns on Investment in Aging Biology"

# Instructions to reproduce results
0. Install the Anaconda distribution of Python (https://www.anaconda.com/download).
1. Clone the repository.
2. Create a conda environment with the required packages:
   ```bash
   conda env create -f environment.yml
   ```
3. Run the bash script to reproduce the results. This computation will take a long time, potentially more than 24 hours. The first script `run_macro_aging.sh` runs the baseline scenario plus the nine main reform scenarios in the paper. Following that, the script executes a high and low sensitivity simulation on each of the 9 scenarios (18 simulations). So the following two scripts represent 28 runs of the macroeconomic model, each of which can take around an hour.
   ```bash
   bash run_macro_aging.sh
    ```
   * If you've cloned the repository, you have all the results from the 28 simulations and can, if you wish, skip rerunning the simulations and just verify that the simulation results can be used to recreate the figures and tables in the paper. To do this, comment out the lines in `run_macro_aging.sh` that run `simulation.py` for the main results and sensitivity results, respectively.
4. All results will be saved in the `ResultsInPaper/macro_aging_results.csv` file and `ResultsInPaper/macro_aging_sensitivity_results.csv`. Individual figures from the paper will be saved in the `ResultsInPaper/figures/` directory.
