# 📈 Stock Return Prediction with PySpark

## 📌 Overview

This project focuses on predicting one-day-ahead stock returns (`Target_Return_1d`) using historical financial time series data. We employ a scalable pipeline powered by PySpark to preprocess, engineer features, perform dimensionality reduction, and train predictive models using sliding window strategies.

## ❓ Statement of the Problem

Accurately forecasting short-term stock returns is notoriously difficult due to market noise and high volatility. Our goal is to build a robust, scalable, and interpretable model to predict `Target_Return_1d` across multiple stocks, using engineered features and time-aware cross-validation.

## 📂 Data Sources and Description
- **Data Sources**:  
- [Kaggle Stock Market Data](https://www.kaggle.com/datasets/paultimothymooney/stock-market-data)
- **Datasets**:  
  - `sliding_time_splits_forbes2000`  
  - `sliding_time_splits_nasdaq`  
  

- **Data Format**:  
  - CSV files: Daily prices (`Open`, `High`, `Low`, `Close`, `Volume`, `Adjusted Close`)  
  - JSON files: Metadata such as currency, instrument type, and exchange info

## 🗂️ Repository Structure

```
├── data/                  # Raw and checkpointed parquet datasets
├── notebooks/             # Exploratory data analysis and prototype modeling
├── src/                   # PySpark pipeline scripts and modules
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
├── output/                # Model outputs and evaluation metrics
├── config/                # Parameters and hyperparameter grids
├── README.md
└── requirements.txt
```

## ⚙️ Approach

We followed a modular pipeline architecture using PySpark:
- Preprocessing with schema enforcement and outlier detection
- Feature engineering (returns, volatility, lag features)
- PCA for dimensionality reduction
- Sliding-window cross-validation (5y train / 1y val / 1y test)
- Modeling with `FMRegressor` and `LinearRegression`, ensembled using time-weighted folds

**Core PySpark Tools Used**:
- `VectorAssembler`, `StandardScaler`, `StringIndexer`, `OneHotEncoder`
- `PCA`, `FMRegressor`, `CrossValidator`, `RegressionEvaluator`

**Parquet checkpointing** was used at preprocessing and fold generation stages to improve reproducibility and reduce recomputation.

## 🗓️ Timeline / Deliverables

- ✅ Week 1–2: Data exploration, schema standardization, outlier detection
- ✅ Week 3–4: Feature engineering, return/volatility calculations
- ✅ Week 5: Sliding window fold generation and preprocessing checkpointing
- ✅ Week 6–7: PCA and model training (FMRegressor + Linear Regression)
- ✅ Week 8: Recency-weighted ensemble, evaluation, and report writing

## 📚 Resources

- [PySpark MLlib Documentation](https://spark.apache.org/docs/latest/ml-guide.html)
- Kaggle Stock Dataset: https://www.kaggle.com/datasets/paultimothymooney/stock-market-data

## 🤝 How to Contribute

This is currently a closed academic project. If you're interested in contributing or adapting this pipeline, feel free to open an issue or contact the authors.

## ▶️ Sample Run / Output

_(To be added...)_
```

---

Let me know once you're ready to plug in sample output, or if you'd like this converted to `.rst` or `.txt` format for broader usage.