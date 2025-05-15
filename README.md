# thesis-electricity-demand-forecasting

# Predicting Long-Term Electricity Demand with Deep Learning Techniques

**Author:** Maria-Antigone Rumpf  
**Institution:** Tilburg University – MSc Data Science & Society  
**Supervisor:** Dr. Giuseppe Cascavilla  
**Timeline:** January – June 2025

## Project Overview

This project explores how deep learning can improve long-term electricity demand forecasting at the national grid level in Germany. The focus is on **comparing the Temporal Fusion Transformer (TFT)** with **Long Short-Term Memory (LSTM)** networks across three time horizons, with a particular emphasis on **30-day-ahead forecasting** — a timeframe underexplored in energy research.

The study introduces **renewable energy availability (wind & solar)** as novel exogenous predictors and evaluates model interpretability using **SHAP values**, to better inform energy planning and policy.

## Research Goals

- **RQ1:** RQ 1: To what extent is the **Temporal Fusion Transformer (TFT)** model able to **outperform LSTM** for _30-day electricity demand forecasting_?
- **SRQ 1.1:** How do **short-term** (day-ahead), **medium-term** (7 days ahead) and **long-term** (30 days ahead) time horizons compare in TFT’s ability to accurately forecast electricity demand?
- **RQ2:** How do the **predictor features** contribute to most accurate long-term electricity demand forecasting, by **comparing SHAP values between TFT and LSTM?**
- **SRQ 2.1:** To what extent do **different time horizons** change what features most accurately predict electricity demand?

## Dataset

- **Source:** ENTSO-E Transparency Platform  
- **Frequency:** 15-minute intervals  
- **Timeframe:** 2015–2020  
- **Rows:** ~200,000  
- **Features:** Electricity load, time-aware calendar features, weather-based features, wind and solar generation

## Methodology (_see full workflow under /reports_)

1. **Data Preparation**
   - Merge raw sources, forward filling weather data
   - Clean data, handle missing values
   - Feature engineering (e.g., cyclic encodings, lag creation)

2. **Exploratory Data Analysis**
   - Demand trends and seasonality
   - Correlation analysis by time horizon
   - Distribution inspection

3. **Modeling**
   - **LSTM**: Classical deep learning model for time-series
   - **TFT**: State-of-the-art model for interpretable, multi-horizon forecasting

4. **Preprocessing**
   - Normalize numeric features
   - Encode cyclical time features
   - Subsample for fast prototyping

5. **Training**
   - Use of GPU servers for high-power computing (HPC), via Linux
   - Hyperparameter tuning (random search)
   - Evaluation on short, medium, and long horizons

6. **Evaluation**
   - Metrics: **MAE** and **MAPE**
   - Interpretability: **SHAP values** for feature importance comparison

## Experimental Design

- Forecasting horizons:
  - **Short-term:** 1 day ahead
  - **Medium-term:** 7 days ahead
  - **Long-term:** 30 days ahead (main focus)
- Models:
  - TFT (multi-horizon)
  - LSTM (retrained for each horizon separately)
- Evaluation:
  - Forecast accuracy (RQ1)
  - Feature importance analysis via SHAP (RQ2)

## Repository Structure

```
thesis-electricity-demand-forecasting/
│
├── data/                     # Raw and processed datasets
│   ├── raw/                  # Original datasets
│   └── processed/            # Cleaned datasets
│
├── notebooks/                # Jupyter notebooks for EDA, cleaning, model prototyping, preprocessing
│
├── reports/                  # Additional resources
├── src/                      # Python scripts
│   ├── main.py               # Main training script, for GPU deployment
│   └── data_processing.py    # Data fetching and preparation
│   └── utils.py              # Helper functions
│
├── .gitignore             
├── README.md
└── requirements.txt              
```

## Clone the repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

## Set up a Python environment
conda create -n demand_forecasting python=3.10
conda activate demand_forecasting

## Install dependencies
pip install -r requirements.txt

## Run the model training
python -m src.main

## 🧠 Key Libraries

- `pytorch-lightning`
- `pytorch-forecasting`
- `tensorflow` / `keras`
- `scikit-learn`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `shap`
