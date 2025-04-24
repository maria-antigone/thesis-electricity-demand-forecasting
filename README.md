# thesis-electricity-demand-forecasting

# ⚡️ Predicting Electricity Demand With Deep Learning

This project forecasts long-term electricity demand in Germany using Temporal Fusion Transformers (TFTs). It compares the deep learning approach against a traditional statistical model (SARIMA) and explores the impact of renewable energy availability as an exogenous feature.

## 📌 Project Goals

- Predict electricity demand 30 days ahead (long-term) using TFT
- Compare performance against SARIMA as a statistical baseline
- Analyze feature importance using SHAP for interpretability
- Understand how renewable energy availability affects demand

## 📁 Repository Structure

```
thesis-electricity-demand-forecasting/
│
├── data/                  # Raw and processed datasets (some ignored by Git)
│   ├── raw/               # Original datasets (ignored)
│   └── processed/         # Cleaned datasets (optional)
│
├── notebooks/             # Jupyter notebooks for EDA and modeling
│   └── cleaning_update.ipynb
│
├── scripts/               # Python scripts for preprocessing or modeling
├── models/                # Saved model files (ignored)
├── reports/               # Visualizations and output summaries
├── src/                   # Source code for reproducibility
│
├── .gitignore             # Specifies files not tracked by Git
├── README.md              # This file
```

## 🧪 How to Run

1. Clone this repository:

git clone https://github.com/maria-antigone/thesis-electricity-demand-forecasting.git cd thesis-electricity-demand-forecasting

2. (Optional) Create and activate a virtual environment

3. Install dependencies:

pip install -r requirements.txt

4. Run the notebook:

jupyter notebook notebooks/cleaning_update.ipynb

## 📊 Datasets

- Electricity demand: [ENTSO-E Transparency Platform](https://transparency.entsoe.eu/)
- Weather data: Open-source datasets for Germany

## 🧹 Data Cleaning Workflow

The full cleaning and merging process is documented in the `notebooks/cleaning_update.ipynb` notebook.

A step-by-step written summary is also available [here](docs/cleaning_workflow.md). *(Coming soon)*

## 📚 Citation

Based on a Master's thesis project in Data Science & Society at Tilburg University (2025).  
Author: Maria-Antigone Rumpf

## 🤝 Contributions

This is a solo thesis project, but suggestions are welcome via GitHub issues or pull requests.

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## 📬 Contact

For questions or collaboration, please reach out via GitHub or email:
