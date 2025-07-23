# Sediment Yield Analysis
Repository for "Machine Learning-Based Sedigraph Reconstruction for Enhanced Sediment Yield Estimation in the Upper Blue Nile Basin" (Worku, 2025, Hydrological Processes submission).

## Repository Structure
- `code/map_plot.py`: Generates Figure 1 (watershed map, JPG).
- `code/precip_eto_plot.py`: Generates Figure 2 (precipitation and evapotranspiration, PNG).
- `code/land_cover_plot.py`: Generates Figure 3 (land cover changes, PNG).
- `code/model_performance.py`: Generates Figures 4 and 5 (model evaluation, PNG).
- `code/feature_importance.py`: Generates Figure 6 (feature importance bar plot, PNG).
- `code/annual_sediment_yield.py`: Generates Figures 7 (annual yields, subfigures a-b, PNG) and 10 (correlations, subfigures a-c, PNG).
- `code/monthly_sediment_yield.py`: Generates Figure 8 (monthly yields, subfigures a-b, PNG).
- `code/seasonal_sediment_yield.py`: Generates Figure 9 (seasonal yields, subfigures a-b, PNG).
- `data/Intermittent_data.csv`: Intermittent data for Gilgel Abay.
- `data/continuous_data.csv`: Continuous data for Gilgel Abay.
- `data/Intermittent_data_gum.csv`: Intermittent data for Gumara.
- `data/continious_data_gum.csv`: Continuous data for Gumara.
- `outputs/`: Contains generated figures (Figure1.jpg, Figure2.png–Figure10.png) and CSVs (e.g., `Gilgel_Abay_Monthly_Data_ha_QRF.csv`, `Gumara_Seasonal_Sediment_Yield_QRF.csv`).
- `requirements.txt`: Python dependencies.
- `LICENSE`: MIT License.

## Data Availability
The Python scripts for generating Figures, raw SSC and hydrological data for Gilgel Abay and Gumara watersheds, and output files (Figure1.png, Figure2–Figure10.png,.svg,.eps, in color and grey scale, and associated CSVs) are available at https://github.com/kindiebitew/sediment-yield-analysis for journal submission.
