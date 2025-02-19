# molecular_docking_prediction
Initial commit with XGBoost model for docking prediction
# Molecular Docking Prediction

This project uses the XGBoost machine learning model to predict molecular docking scores. The model is trained and tested with data from **docking_simulation.csv**. The accuracy of the model is evaluated using the MAE (Mean Absolute Error) metric.

## Requirements
- pandas
- numpy
- xgboost
- scikit-learn
- matplotlib
- seaborn

## Usage
1. Load the data and start model training.
2. Train the regression model using XGBoost.
3. Evaluate the model’s accuracy with the MAE metric.

## Model Performance
Model MAE: `0.231`

![Actual vs Predicted Plot](results/actual_vs_predicted.png)
