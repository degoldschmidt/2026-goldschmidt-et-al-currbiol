# Goldschmidt et al. (2026) - Code for Figure Generation

Code and analysis scripts for generating figure panels from "Recent experience and internal state shape local search strategies in flies" (*Current Biology*, 2026).

## Overview

This repository contains Jupyter notebooks and Python scripts for reproducing all figure panels (Figures 1-3, 5-7, and Supplementary Figures, as well as Data S1).

**Note:** Figure 4 code is located in a separate repository.

## Repository Structure

- **`sourcecode_figX.ipynb`** - Jupyter notebooks for generating figure panels (Figures 1, 2, 3, 5, 6, 7)
- **`src/`** - Python modules with core analysis functions:
  - `helper.py` - Utility functions
  - `makedataframe.py` - Data frame construction and processing
  - `per_fly.py` - Per-fly analysis
  - `per_trips.py` - Per-trip analysis
  - `viz.py` - Visualization utilities
  - `interface.py` - Data interface functions
- **`plt/`** - Output directory for generated figures and source data
- **`dat/`** - Processed data directory

## Requirements

- Python 3.7+
- pandas, numpy, matplotlib, seaborn, scipy
- statsmodels
- scikit-learn
- tqdm

## Usage

Run the Jupyter notebooks to regenerate figures:

```bash
jupyter notebook sourcecode_fig1.ipynb
jupyter notebook sourcecode_fig2.ipynb
# ... etc for other figures
```

Notebooks load pre-processed tracking data from feather and YAML files (not included in this repository due to file size).

## Data

Raw per-frame tracking data and metadata must be provided separately. Expected data format:
- `.feather` files with per-frame tracking data
- `.yaml` files with experiment metadata

Update the `rootDir` variable in each notebook to point to your data directory.

## Figures Generated

- **Figures 1, S1** - Loop search strategies in control flies
- **Figures 2, S2** - Trip classification and trip-level search dynamics
- **Figures 3, S3** - Locomotor parameters and search behavior
- **Figures 5, S5** - EPG silencing effects on search
- **Figures 6, S6** - Sensory receptor mutant effects on search
- **Figures 7, S7** - Optogenetic validation of sensory inputs
- **Data S1** - Supplementary analyses and statistics

## Citation

If you use this code, please cite:

Goldschmidt et al. (2026). Recent experience and internal state shape local search strategies in flies. *Current Biology*.

