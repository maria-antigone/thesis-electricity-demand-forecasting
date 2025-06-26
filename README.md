# thesis-electricity-demand-forecasting

# Predicting Long-Term Electricity Demand with Deep Learning Techniques

**Author:** Maria-Antigone Rumpf  
**Institution:** Tilburg University – MSc Data Science & Society  
**Supervisor:** Dr. Giuseppe Cascavilla  
**Timeline:** January – June 2025

## Project Overview

This thesis explores the use of deep learning to forecast electricity demand at high frequency (15-minute intervals) on a national scale. It compares the performance and interpretability of two models — **Temporal Fusion Transformer (TFT)** and **Long Short-Term Memory (LSTM)** — with a focus on **30-day-ahead forecasting**, a challenging but underexplored task in the energy domain.

The study incorporates **renewable energy availability (wind and solar generation)** as exogenous inputs and uses TFT’s **Variable Selection Networks (VSNs)** to evaluate which features contribute most to model predictions at different time horizons.

## Research Questions

- **RQ1:** To what extent does the **TFT model outperform LSTM** for 30-day electricity demand forecasting?
  - **SRQ1.1:** How does TFT’s forecasting accuracy vary across **1-day**, **7-day**, and **30-day** horizons?

- **RQ2:** How do **input features** contribute to long-term forecasting accuracy, as revealed by TFT’s **Variable Selection Networks**?
  - **SRQ2.1:** How does **feature importance change across forecasting horizons**?

## Dataset

- **Source:** Open Power System Data (via ENTSO-E Transparency Platform)
- **Frequency:** 15-minute intervals  
- **Period:** 2015–2020  
- **Rows:** ~175,000  
- **Features:**  
  - Electricity load  
  - Temperature and calendar features  
  - Wind and solar generation  

## Methodology (_see workflow in `/reports`_)

1. **Data Preparation**
   - Merge datasets and forward-fill weather data
   - Drop low-quality columns and impute missing values
   - Engineer time features (e.g., hour, weekend, daylight) with cyclical encoding

2. **Exploratory Data Analysis**
   - Visualize seasonality and feature distributions
   - Analyze non-linear relationships and correlation structures

3. **Preprocessing**
   - Normalize numerical variables with MinMaxScaler
   - Sequence and batch creation per model and horizon
   - Adjusted input lengths based on GPU constraints

4. **Modeling**
   - **LSTM**: Standard sequence model, one model per horizon  
   - **TFT**: Attention-based, interpretable multi-horizon model

5. **Training**
   - Performed on Tilburg University GPU cluster
   - Used **Optuna** for hyperparameter tuning (short horizon only)
   - Horizon-specific models trained with early stopping

6. **Evaluation**
   - Metrics: **MAE**, **RMSE**, **MAPE**
   - Interpretability: **TFT’s Variable Selection Networks (VSNs)**

## Experimental Setup

- **Horizons:**  
  - Short-term: 1-day-ahead (96 steps)  
  - Medium-term: 7-days-ahead (672 steps)  
  - Long-term: 30-days-ahead (2880 steps)

- **Model Setup:**  
  - TFT (multi-horizon model trained per horizon)  
  - LSTM (baseline trained per horizon)

- **Interpretability:**  
  - Cross-horizon comparison of feature relevance via VSNs

## Repository Structure

```
thesis-electricity-demand-forecasting/
│
├── data/                     # Raw and processed datasets
│   ├── raw/                  # Original datasets
│   └── processed/            # Cleaned datasets
│
├── notebooks/                # Jupyter notebooks for EDA, cleaning, model prototyping, preprocessing, visualizations
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

## Setup Instructions

```bash
# Clone the repository
git clone https://github.com/your-username/thesis-electricity-demand-forecasting.git
cd thesis-electricity-demand-forecasting

# Set up a Python environment
conda create -n demand_forecasting python=3.10
conda activate demand_forecasting

# Install dependencies
pip install -r requirements.txt

# Run the model training
python -m src.main
## 🧠 Key Libraries

- `pytorch-lightning`
- `pytorch-forecasting`
- `optuna`
- `tensorflow` / `keras`
- `scikit-learn`
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
