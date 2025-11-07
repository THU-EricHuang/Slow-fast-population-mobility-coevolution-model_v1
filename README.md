# Slow-fast-population-mobility-coevolution-model_v1
Thank you for reviewing our manuscript "Population-mobility coevolution drives the emergence of spatial heterogeneity in cities”.  The repository contains the source code. The source code is implemented in Python.

## Environment Requirements

This code is developed based on **Python 3.9+**. It is highly recommended to use [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) to create an independent virtual environment.

**Main Dependencies (Recommended Versions):**
* `python ~= 3.9`
* `numpy ~= 1.21.0`
* `pandas ~= 1.4.0`
* `scipy ~= 1.8.0`
* `scikit-learn ~= 1.1.0`
* `geopandas ~= 0.11.0`
* `matplotlib ~= 3.5.0`
* `seaborn ~= 0.12.0`
* `tqdm ~= 4.64.0`

### Installation Guide

1.  Clone this repository:
    ```bash
    git clone [https://github.com/THU-EricHuang/Slow-fast-population-mobility-coevolution-model_v1.git](https://github.com/THU-EricHuang/Slow-fast-population-mobility-coevolution-model_v1.git)
    cd Slow-fast-population-mobility-coevolution-model_v1
    ```

2.  Create and activate the Anaconda environment (Recommended):
    ```bash
    conda create -n mobility_coevol python=3.9
    conda activate mobility_coevol
    ```

3.  Install the required packages. You can install them manually using `pip`:
    ```bash
    pip install numpy~=1.21.0 pandas~=1.4.0 scipy~=1.8.0 scikit-learn~=1.1.0 geopandas~=0.11.0 matplotlib~=3.5.0 seaborn~=0.12.0 tqdm~=4.64.0
    ```

---

## How to Run

The workflow is divided into data preprocessing and main model execution.

### 1. Data Preprocessing

Before running the main model, a series of data preprocessing steps must be executed. These scripts (`pre1` to `pre7`) are used to process the raw data (e.g., CBG, OD, POI) and convert it into the grid-based inputs required by the model.

**Please execute the following scripts in order:**
```bash
python pre1_US_CBG_extract.py
python pre2_US_CBG_od.py
python pre3_US_grid_pop.py
python pre4_US_cbg2grid_od.py
python pre5_Overturemaps_POI_download.py
python pre6_Cal_functionsimilarity.py
python pre7_Cal_distance.py
```

Note: Ensure that all necessary raw data files (e.g., mobility data, Census data) are placed in the raw data folder before starting.

### 2. Run the Main Model
After data preprocessing is complete, run the main simulation:
```bash
python run_main.py
```
The simulation results will be saved to the Data&Result_US_SantaClaraCounty directory or the specified output folder.

## File Structure
```bash
.
├── Data&Result_US_SantaClaraCounty/
│   └── (Used to store preprocessed data, intermediate results, and final model outputs)
├── raw data/
│   └── (Used to store all raw data, e.g., Census CBG data, etc. Need to be download with the open link provided by U.S. government. OD data from large company should be paid in official website.)
├── utils/
│   └── (Contains utility functions and helper scripts required for the model)
│
├── pre1_US_CBG_extract.py         # Preprocessing Step 1: Extract US CBG data
├── pre2_US_CBG_od.py              # Preprocessing Step 2: Process CBG-to-CBG OD (Origin-Destination) data
├── pre3_US_grid_pop.py            # Preprocessing Step 3: Grid population data
├── pre4_US_cbg2grid_od.py         # Preprocessing Step 4: Map CBG OD data to the grid
├── pre5_Overturemaps_POI_download.py # Preprocessing Step 5: Download Overturemaps POI data
├── pre6_Cal_functionsimilarity.py # Preprocessing Step 6: Calculate functional similarity based on POI
├── pre7_Cal_distance.py           # Preprocessing Step 7: Calculate distances between grids
│
├── run_main.py                    # Main script: Run the co-evolution model
├── Final_twoDistributionPlotter_ALL_clean.py # Plotting script: Generate final distribution plots
│
└── README.md                      # This file
```

