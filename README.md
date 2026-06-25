# Zelbytes Yield Forecast

## Problem Statement

The objective of this project is to develop a machine learning solution for predicting daily mushroom yield (kg) in a climate-controlled polyhouse using environmental sensor readings such as temperature, humidity, and CO₂ concentration.

## Environment Setup

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```
## Live Demo

Streamlit Application:



### Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

### Model

* Champion Model: Tuned Random Forest Regressor
* Features: Temperature, Humidity, CO₂
* Metric Used: MAE, RMSE, R²

## Folder Structure

```text
zelbytes-yield-forecast/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── src/
├── models/
├── requirements.txt
├── README.md
└── .gitignore
```

## Dataset Columns

| Column | Description | Unit |
|----------|------------|------|
| timestamp | Date and time of sensor reading | Datetime |
| temperature_c | Polyhouse temperature | °C |
| humidity_pct | Relative humidity inside polyhouse | % |
| co2_ppm | Carbon dioxide concentration | ppm |
| yield_kg | Mushroom yield harvested | kg |

### Dataset Notes

- Temperature is measured in degrees Celsius.
- Humidity is measured as relative humidity percentage.
- CO₂ concentration is measured in parts per million (ppm).
- Yield represents the harvested mushroom weight in kilograms.

## Data Quality Analysis

- Generated descriptive statistics using Pandas.
- Calculated Coefficient of Variation (CV) for all numerical features.
- Verified data consistency after cleaning.

## Exploratory Data Analysis

- Created correlation heatmaps and scatter plots.
- Temperature showed the strongest positive relationship with yield.
- Humidity exhibited a moderate negative correlation with yield.
- CO₂ had a relatively weak impact on yield.


##Feature Engineering

| Feature         | Description                               |
| --------------- | ----------------------------------------- |
| temperature     | Polyhouse temperature reading             |
| humidity        | Relative humidity reading                 |
| CO2             | Carbon dioxide concentration              |
| Min-Max Scaling | Scales feature values to the range [0, 1] |

- Selected temperature, humidity, and CO₂ as baseline predictive features.
- Applied Min-Max Scaling to normalize feature values.
- Scaling improves consistency across features with different ranges.
- Target variable (yield) was kept unchanged.
- Fitted scaler was saved as minmax_scaler.joblib for future inference.

## Data Split

| Parameter          | Value                               |
| ------------------ | ----------------------------------- |
| Split Method       | Chronological (Time-Based)          |
| Training Data      | 80%                                 |
| Testing Data       | 20%                                 |
| Sort Order         | Timestamp Ascending                 |
| Leakage Prevention | Scaler fitted on training data only |

- Dataset was sorted by timestamp before splitting.
- Earlier observations were used for training and later observations for testing.
- A chronological split was used to simulate real-world forecasting conditions.
- Data leakage was prevented by fitting the Min-Max Scaler only on training data.
- Training and testing datasets were saved as:


### Model Training
| Model               | MAE (kg)  | RMSE (kg) | R²        |
| ------------------- | --------- | --------- | --------- |
| Linear Regression   | 0.09      | 0.12      | 0.696     |
| Random Forest       | 0.09      | 0.12      | 0.707     |
| Tuned Random Forest | 0.081     | 0.111     | 0.737     |

- Trained and evaluated Linear Regression and Random Forest models for mushroom yield prediction.
- Performance was measured using MAE, RMSE, and R² metrics.
- TimeSeriesSplit cross-validation was used to assess model stability.
- GridSearchCV was applied to optimize Random Forest hyperparameters.
- The Tuned Random Forest achieved the best performance and was selected as the final model.
# Best Hyperparameters
| Parameter        | Value |
| ---------------- | ----- |
| n_estimators     | 200   |
| max_depth        | None  |
| min_samples_leaf | 3     |

