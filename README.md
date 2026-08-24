# Bike Rental Count Prediction & Comprehensive Analysis

## 📌 Project Overview

This project is an end-to-end **Machine Learning regression and time-series analysis** solution for predicting daily bike rental demand using the Capital Bikeshare dataset.

The project covers exploratory data analysis, statistical hypothesis testing, feature engineering, leakage prevention, preprocessing, model comparison, hyperparameter tuning, model evaluation, explainability, and model serialization for deployment.

**Project:** PRCP-1018: Bike Rental Count Prediction & Comprehensive Analysis  
**Project Type:** Regression / Time Series Analysis  
**Contribution:** Individual  
**Author:** Hemaa Shri G

---

## 🎯 Objective

The primary objective is to predict the **daily total number of bike rentals (`cnt`)** using calendar, weather, and recent-demand information.

The analysis also investigates the business factors that influence rental demand, including:

- Year-over-year growth
- Seasonal demand
- Weather conditions
- Temperature and humidity
- Recent rental demand
- Long-term demand trend
- Casual vs. registered rider behaviour

---

## 📊 Dataset

The project uses the Capital Bikeshare daily and hourly datasets:

- `day.csv`
- `hour.csv`

The daily dataset contains **731 records**, covering **January 1, 2011 to December 31, 2012**, with 16 original columns.

The target variable is:

```text
cnt
```

which represents the total number of bike rentals for a day.

### Important original features

- `dteday` — date
- `season` — season category
- `yr` — year
- `mnth` — month
- `holiday` — holiday indicator
- `weekday` — day of week
- `workingday` — working-day indicator
- `weathersit` — weather situation
- `temp` — normalized temperature
- `atemp` — normalized feeling temperature
- `hum` — normalized humidity
- `windspeed` — normalized wind speed
- `casual` — casual-user rentals
- `registered` — registered-user rentals
- `cnt` — total rentals

---

## 🔍 Exploratory Data Analysis

The project investigates several important patterns in the dataset.

### Key findings

- Average daily rentals increased by approximately **64.4%** from 2011 to 2012.
- Temperature has a positive relationship with demand (**r ≈ 0.63**).
- The 7-day rolling rental average has a very strong relationship with current demand (**r ≈ 0.89**).
- Yesterday's rental count (`cnt_lag1`) also provides strong predictive information (**r ≈ 0.85**).
- Poor weather conditions substantially reduce rental demand.
- Registered users account for approximately **81.8%** of total rentals, while casual users account for approximately **18.2%**.

These findings support the use of both **time-based features and recent-demand features** in the predictive models.

---

## 🧪 Hypothesis Testing

Three statistical hypotheses were tested at a significance level of **α = 0.05**.

### 1. Year-over-Year Rental Expansion

**Test:** Welch's Two-Sample t-test

The analysis tested whether average daily rentals in 2012 were significantly higher than in 2011.

**Conclusion:** The increase was statistically significant (**p < 0.001**).

### 2. Seasonal Demand Variation

**Test:** One-Way ANOVA

The analysis tested whether average daily rentals differ across the four seasons.

**Conclusion:** Rental demand differs significantly across seasons (**p < 0.001**).

### 3. Weather Situation Impact

**Test:** Kruskal-Wallis H-test

The analysis tested whether weather situation significantly affects daily rental demand.

**Conclusion:** Weather conditions have a statistically significant effect on rentals (**p < 0.001**).

---

## ⚙️ Feature Engineering & Preprocessing

Several features were engineered to capture time and demand behaviour.

### Engineered features

- `trend_days` — captures long-term business growth
- `sin_mnth` — cyclical representation of month
- `cos_mnth` — cyclical representation of month
- `cnt_lag1` — previous day's rental count
- `cnt_lag7` — rental count seven days earlier
- `cnt_7d_mean` — rolling seven-day average demand

### Target transformation

The target `cnt` is right-skewed, so the project applies:

```python
log1p(cnt)
```

during model training to stabilize variance and reduce the influence of extreme values.

### Scaling

`StandardScaler` is used where required, and the scaler is fitted only on the training data to prevent data leakage.

### Categorical encoding

- Tree-based models use the categorical codes directly.
- Linear models use one-hot encoding for categorical variables.

### Outlier handling

The extreme low-rental day associated with **Hurricane Sandy (October 29, 2012)** is retained because it is a real business event rather than an invalid observation.

Instead of deleting the record, the project uses the `log1p(cnt)` transformation and evaluates models using robust percentage metrics such as WAPE and SMAPE.

---

## 🚨 Data Leakage Prevention

A major challenge in this project was the relationship:

```text
casual + registered = cnt
```

Using `casual` and `registered` as predictors would leak the target directly into the model.

Therefore, both columns were explicitly removed from the final modelling features.

The project also removes:

- `atemp` — due to near-perfect correlation with `temp` (r ≈ 0.99)
- `instant` — treated as a row identifier
- Raw `dteday` string — replaced with meaningful time-derived features

This helps make the model more realistic for future unseen predictions.

---

## ⏳ Time-Series Train/Test Split

Because this is time-dependent data, the project uses an **80:20 chronological split** rather than randomly shuffling observations.

- Training: first **584 days**
- Testing: last **147 days**

This prevents future information from leaking into the training set and provides a more realistic evaluation of future demand prediction.

---

## 🤖 Machine Learning Models

A total of **13 regression algorithms** were trained and evaluated using the same chronological train/test split.

Models include:

1. Gradient Boosting
2. XGBoost
3. LightGBM
4. Extra Trees
5. CatBoost
6. Linear Regression
7. Ridge Regression
8. Gaussian Process
9. Bayesian Ridge
10. Random Forest
11. ElasticNet
12. Decision Tree
13. Lasso Regression

Hyperparameter tuning was performed for relevant models using cross-validation and grid/randomized search.

---

## 📈 Model Comparison

All models were evaluated using:

- R²
- Adjusted R²
- RMSE
- MAE
- Median Absolute Error
- WAPE
- SMAPE

### Final benchmark

| Rank | Model | R² | Adjusted R² | RMSE | MAE | WAPE |
|---:|---|---:|---:|---:|---:|---:|
| 1 | **Gradient Boosting** | **0.791** | **0.766** | **856** | **636** | **10.79%** |
| 2 | XGBoost | 0.777 | 0.749 | 886 | 678 | 11.50% |
| 3 | LightGBM | 0.754 | 0.724 | 929 | 677 | 11.48% |
| 4 | Extra Trees | 0.754 | 0.723 | 930 | 731 | 12.40% |
| 5 | CatBoost | 0.733 | 0.700 | 968 | 792 | 13.42% |
| 6 | Linear Regression | 0.724 | 0.690 | 985 | 790 | 13.39% |
| 7 | Ridge Regression | 0.723 | 0.689 | 986 | 755 | 12.79% |
| 8 | Gaussian Process | 0.720 | 0.685 | 992 | 784 | 13.30% |
| 9 | Bayesian Ridge | 0.720 | 0.685 | 992 | 784 | 13.30% |
| 10 | Random Forest | 0.713 | 0.678 | 1005 | 729 | 12.37% |
| 11 | ElasticNet | 0.675 | 0.635 | 1069 | 818 | 13.87% |
| 12 | Decision Tree | 0.577 | 0.525 | 1219 | 846 | 14.34% |
| 13 | Lasso Regression | 0.573 | 0.520 | — | — | — |

> The values above are the benchmark results recorded in the project notebook.

---

## 🏆 Final Model

### Gradient Boosting Regressor

The final recommended model is **Gradient Boosting Regressor**.

It achieved:

- **R²:** 0.7913
- **RMSE:** approximately 856 bikes/day
- **WAPE:** approximately 10.79%

The model explains approximately **79% of the variation** in daily rental demand on the held-out test set.

### Why Gradient Boosting?

Gradient Boosting outperformed the other tested algorithms on the final benchmark and provided the best combination of predictive accuracy and business-relevant error metrics.

---

## 🔎 Feature Importance

The most important predictive features identified by the winning model include:

1. `cnt_7d_mean`
2. `cnt_lag1`
3. `trend_days`
4. `temp`
5. `hum`

The first two features demonstrate that **recent rental demand is the strongest signal for near-future demand**, while `trend_days` captures long-term business growth.

Temperature and humidity provide additional information about weather-driven demand changes.

---

## 💼 Business Insights

The analysis provides several actionable insights:

- Demand increased significantly from 2011 to 2012.
- Summer months produce stronger demand.
- Poor weather can substantially reduce rentals.
- Registered riders form the majority of the customer base.
- The summer increase in casual riders creates an opportunity for membership-conversion campaigns.
- Recent demand is highly predictive of near-future demand, making rolling and lag features valuable for operational planning.

A more accurate demand forecast can support:

- Bike availability planning
- Station stock management
- Staffing decisions
- Seasonal promotions
- Membership campaigns
- Operational resource allocation

---

## 🚀 Model Deployment Preparation

The project includes model serialization using `joblib`.

The best model is saved as:

```text
bike_rental_best_model.pkl
```

The scaler is saved as:

```text
scaler.pkl
```

The notebook also reloads the saved model and scaler and performs an unseen-data sanity check.

---

## 📁 Repository Structure

A recommended GitHub repository structure is:

```text
Bike-Rental-Count-Prediction/
│
├── README.md
├── .gitignore
├── LICENSE
├── PRCP_1018_BikeRental_Final_Submission(1).ipynb
│
├── data/
│   ├── day.csv
│   └── hour.csv
│
└── models/
    ├── bike_rental_best_model.pkl
    └── scaler.pkl
```

> Add the `models/` directory only if you decide to upload the serialized model files. Keep dataset and model filenames consistent with your actual repository.

---

## 🛠️ Technologies & Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- SciPy
- Scikit-learn
- XGBoost
- LightGBM
- CatBoost
- Statsmodels
- Joblib
- Jupyter Notebook

---

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd Bike-Rental-Count-Prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the datasets

Place:

```text
day.csv
hour.csv
```

inside the `data/` directory, or update the notebook paths if your dataset location is different.

### 4. Open the notebook

```bash
jupyter notebook "PRCP_1018_BikeRental_Final_Submission(1).ipynb"
```

Run the notebook cells sequentially.

---

## 🔮 Future Improvements

Possible future enhancements include:

- Deploying the model as a web application or API
- Creating an interactive bike-demand dashboard
- Adding live weather information
- Incorporating additional external demand signals
- Retraining the model with newer rental data
- Monitoring model drift after deployment
- Creating automated daily demand forecasts

---

## 👤 Author

**Hemaa Shri G**

Machine Learning / Data Science Project

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for details.

---

⭐ If you find this project useful, consider giving the repository a star!
