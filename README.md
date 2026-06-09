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

### Run Smoke Test

```bash
python src/smoke_test.py
```

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