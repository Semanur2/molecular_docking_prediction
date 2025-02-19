import pandas as pd
from sklearn.model_selection import train_test_split

# Load data
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

# Clean the data and select features
def preprocess_data(df):
    df_cleaned = df.dropna()  # Remove rows with missing values
    X = df_cleaned.drop(columns=["Ki(M)", "log(Ki)", "pKi"])  # Remove unnecessary columns
    y = df_cleaned["log(Ki)"]  # Target column for prediction
    return X, y

# Split data into training and testing sets
def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test
