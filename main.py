import pandas as pd
import matplotlib.pyplot as plt  # Import matplotlib for plotting
import seaborn as sns  # Import seaborn for better visualization
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

# Read the CSV file
file_path = "C:/Users/139sa/Desktop/blac/docking_simulation.csv"
df = pd.read_csv(file_path)

# Display the first 5 rows to inspect the dataset
print(df.head())

# Check for missing values in the dataset
print(df.isnull().sum())

# Set the target variable 'Affinity(kcal/mol)'
y = df['Affinity(kcal/mol)']

# Select features for the model
X = df[['Torsions', 'Gauss 1', 'Gauss 2', 'Pose Number', 'Hydrophobic', 'Repulsion', 'Hydrogen', 'Torsional']]

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the XGBoost model
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Calculate Mean Absolute Error (MAE) for the predictions
mae = mean_absolute_error(y_test, y_pred)
print(f"XGBoost Model MAE: {mae}")

# Hyperparameter tuning with GridSearchCV
xg_reg = xgb.XGBRegressor(objective='reg:squarederror')

# Define the parameter grid to search over
param_grid = {
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 2, 3],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.7, 0.8, 1.0]
}

# Perform grid search to optimize hyperparameters
grid_search = GridSearchCV(estimator=xg_reg, param_grid=param_grid, cv=3, scoring='neg_mean_absolute_error', verbose=1)
grid_search.fit(X_train, y_train)

# Print the best parameters found by grid search
print(f"Best parameters: {grid_search.best_params_}")

# Use the best model found by grid search for prediction
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Recalculate MAE for the best model
mae = mean_absolute_error(y_test, y_pred)
print(f"Best model MAE: {mae}")

# Save the plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual Values (Y Test)")
plt.ylabel("Predicted Values (Y Pred)")
plt.title("XGBoost Model - Actual vs Predicted Values")

# Add a red line representing perfect predictions
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)

# Save the plot to a file
plt.savefig("results/actual_vs_predicted.png")

# Display the plot
plt.show()
