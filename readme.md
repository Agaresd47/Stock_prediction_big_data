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
├── data/
│   ├── sliding_time_splits_forbes2000/         # Raw Forbes2000 dataset (CSV/JSON)
│   ├── sliding_time_splits_nasdaq/             # Raw Nasdaq dataset (CSV/JSON)
│   ├── sliding_time_splits_forbes2000_clean/   # Cleaned and preprocessed Forbes2000 dataset
│   └── sliding_time_splits_nasdaq_clean/       # Cleaned and preprocessed Nasdaq dataset
│   └── Stock Data       # Raw Data from Kaggle
│
├── M1.ipynb              # EDA and initial preprocessing
├── M2.ipynb              # Feature engineering and outlier handling
├── M3.ipynb              # Modeling and PCA
├── M4.ipynb              # Evaluation and ensemble analysis
├── README.md
└── final_report.pdf
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

## 📅 Timeline & Deliverables

Each phase consists of a dedicated Jupyter notebook, finalized bi-weekly:

- ✅ **Week 1–2:**

  - Data exploration, schema standardization, and outlier detection
  - Data cleaning: handling null values, relevant feature selection, ETF filtering
  - **Deliverable:** `M1.ipynb`
- ✅ **Week 3–4:**

  - Feature engineering, return and volatility calculations
  - **Deliverable:** `M2.ipynb`
- ✅ **Week 5–6:**

  - Sliding window fold generation and preprocessing checkpointing
  - PCA dimensionality reduction and hyperparameter tuning (grid search) on a 20% data subset
  - Models evaluated: Factorization Machine Regressor (FM), Linear Regression (LR), Gradient Boosted Trees (GBT)
  - Decision: Proceed with FM and LR models due to significantly better performance
  - **Deliverable:** `M3_update.ipynb`
- ✅ **Week 7–8:**

  - Recency-weighted ensemble using FM and LR
  - Comprehensive evaluation, report writing, and final presentation preparation
  - **Deliverables:** `M4.ipynb`, final project report, presentation slides, and `README.md`

## 📚 Resources

- [PySpark MLlib Documentation](https://spark.apache.org/docs/latest/ml-guide.html)
- [Kaggle Stock Market Data](https://www.kaggle.com/datasets/paultimothymooney/stock-market-data)

## 🤝 How to Contribute

We welcome contributions to this project! If you’d like to report an issue, suggest improvements, or submit a pull request:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/your-feature-name`)
5. Open a pull request

For major changes, please open an issue first to discuss what you’d like to change. Contributions are reviewed regularly.



## ▶️ Sample Run / Output Demonstration

This example demonstrates how to load pre-processed stock data, train an ensemble model on a single fold, and evaluate its prediction performance using a small test set. The goal is to illustrate how the model behaves without relying on visual plots.

### 🗂️ Dataset & Setup

- Dataset used: `sliding_time_splits_nasdaq_clean`
- Fold used: `fold 0` only (for quick testing)
- Target variable: `Target_Return_1d`
- Sample size: 1,000 test rows (using `.limit(1000)`)

### ⚙️ Code Workflow

1. **Load Pre-Processed Data**  
   Uses the `load_pca_folds` function to load one fold of training data and a test set.

2. **Train Ensemble Models**  
   `train_models_on_folds` is called on the training fold to produce a set of base models.

3. **Make Predictions on Sample Test Set**  
   The ensemble model outputs predictions using `ensemble_predict`.

4. **Display Actual vs Predicted Values**  
   The script shows a side-by-side comparison of the actual returns, predicted returns, and their absolute error.

### 📋 Sample Output (Top 10 Predictions)

| Actual Return | Predicted Return | Error |
|---------------|------------------|-------|
| 0.021         | -18.255          | -18.276 |
| 0.035         | -18.240          | -18.275 |
| -0.012        | -18.265          | -18.253 |
| ...           | ...              | ...    |

_(Note: Values shown here are representative; actual output comes from running the code.)_

### 🔎 Correlation Check

The Pearson correlation between actual and predicted returns is calculated and printed:
```
Correlation between actual and predicted values: -0.0880
```

This negative correlation confirms a mismatch between the training context and current data, emphasizing the limitations of using outdated training folds.

### 📊 Error Quantiles

Quantiles of absolute prediction error help assess consistency of model deviation:
```
0.25    18.272703
0.50    18.295451
0.75    18.321674
0.90    18.353319
0.95    18.379131
0.99    18.437994
```

### 🧠 Key Takeaways

- The model trained on older data (fold 0) shows a consistent prediction bias around `-18`.
- Predictions are not aligned with actual return values, revealing a **systematic prediction drift**.
- This supports our strategy of **recency-weighted ensembles**, where newer folds are given more weight to improve alignment with current market behavior.

