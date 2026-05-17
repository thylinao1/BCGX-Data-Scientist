# Data

The raw and processed PowerCo datasets are not committed to this repository.
The data originates from the BCG X Open Data Science Job Simulation on
[Forage](https://www.theforage.com/) and the platform restricts
redistribution.

To reproduce the analysis, place the following files in this directory:

| File | Source | Used by |
|------|--------|---------|
| `client_data.csv` | Forage task package | `notebooks/01_eda.ipynb` |
| `price_data.csv` | Forage task package | `notebooks/01_eda.ipynb` |
| `clean_data_after_eda.csv` | produced by `01_eda.ipynb` | `notebooks/02_feature_engineering.ipynb` |
| `data_for_predictions.csv` | produced by `02_feature_engineering.ipynb` | `notebooks/03_modelling.ipynb` |

After downloading the two raw files, run the notebooks in order to regenerate
the intermediate files.

All paths in the codebase are relative; pass a custom `data_dir` to the
loaders in `src/data_loading.py` if you store the data elsewhere.
