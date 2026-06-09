# Cleaning Log

## Dataset

polyhouse_sensors.csv

## Missing Values Before Cleaning

| Column        | Null Count |
| ------------- | ---------- |
| temperature_c | 3          |
| humidity_pct  | 3          |
| co2_ppm       | 4          |
| yield_kg      | 7          |

## Cleaning Strategy

### temperature_c

Missing values replaced using the median temperature because missing readings are likely caused by temporary sensor failures.

### humidity_pct

Missing values replaced using the median humidity value to preserve normal environmental conditions.

### co2_ppm

Missing values replaced using the median CO₂ concentration because missing values are likely due to logging interruptions.

### yield_kg

Rows with missing yield values were removed. Yield is the target variable and should not be imputed because it may introduce bias into machine learning models.

## Duplicate Handling

Duplicate rows were checked and removed where necessary.

## Validity Rules Applied

* Humidity must be between 0% and 100%.
* CO₂ concentration must be greater than 0 ppm.
* Temperature must be within a realistic range (0–50°C).

## Missing Values After Cleaning

All remaining columns contain zero missing values.

## Output

Cleaned dataset exported to:

data/processed/02_cleaned.parquet
