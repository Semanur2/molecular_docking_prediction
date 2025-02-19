import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from preprocessing import load_data, preprocess_data, split_data

# Load and preprocess the data
df = load_data("data/docking_simulation.csv")
X, y = preprocess_data(df)
X_train, X_test, y_train, y_test = split_data(X, y)

# Create the XGBoost model
model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate model performance
mae = mean_absolute_error(y_test, y_pred)
print(f"XGBoost Model MAE: {mae}")

# Visualize the results
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Values (Y Test)")
plt.ylabel("Predicted Values (Y Pred)")
plt.title("XGBoost Model - Actual vs Predicted Values")
plt.show()
