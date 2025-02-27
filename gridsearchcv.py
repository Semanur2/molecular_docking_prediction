import pandas as pd
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error

# Dosya yolunu belirtin ve veri çerçevesini yükleyin
file_path = "C:/Users/139sa/Desktop/blac/docking_simulation.csv"  # Yolu kendi dosyanıza göre güncelleyin
df = pd.read_csv(file_path)

# Eksik değerleri doldur
df.fillna(df.mean(numeric_only=True), inplace=True)  # Sayısal sütunları ortalama ile doldur
df.fillna(df.mode().iloc[0], inplace=True)  # Kategorik sütunları mod ile doldur

# Özellikleri ve hedefi seçin
X = df[['Torsions', 'Gauss 1', 'Gauss 2', 'Pose Number', 'Hydrophobic', 'Repulsion', 'Hydrogen', 'Torsional']]
y = df['Affinity(kcal/mol)']

# Eğitim ve test verilerine ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modeli tanımla
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# Hiperparametrelerin aralığını belirle
param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_child_weight': [1, 3, 5],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'gamma': [0, 0.1, 0.2]
}

# GridSearchCV uygulayalım
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_absolute_error', verbose=2, n_jobs=-1)

# Eğitim verileriyle grid search yap
grid_search.fit(X_train, y_train)

# En iyi parametreleri yazdır
print(f"En iyi hiperparametreler: {grid_search.best_params_}")

# En iyi model
best_model = grid_search.best_estimator_

# Test verileriyle tahmin yap
y_pred_best = best_model.predict(X_test)

# Yeni modelin hata hesaplaması
mae_best = mean_absolute_error(y_test, y_pred_best)
print(f"Yeni Modelin Ortalama Mutlak Hatası (MAE): {mae_best}")

# Ligand bazlı tahminleri ekleyelim
df['Predicted Affinity'] = best_model.predict(X)  # En iyi model ile tahminler

# Top 10 ligandı sıralayalım
top_10_ligands = df.drop_duplicates(subset=['Ligand']).nsmallest(10, 'Predicted Affinity')

# Top 10 ligandı yazdıralım
print("\nTop 10 Predicted Ligands and Their Affinities:")
for index, row in top_10_ligands.iterrows():
    print(f"Ligand: {row['Ligand']}, Predicted Affinity: {row['Predicted Affinity']:.2f} kcal/mol")
