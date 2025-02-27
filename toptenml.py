import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Dosya yolunu belirtin ve veri çerçevesini yükleyin
file_path = "C:/Users/139sa/Desktop/blac/docking_simulation.csv"
df = pd.read_csv(file_path)

# Eksik değerleri doldur
df.fillna(df.mean(numeric_only=True), inplace=True)  # Sayısal sütunları ortalama ile doldur
df.fillna(df.mode().iloc[0], inplace=True)  # Kategorik sütunları mod ile doldur

# Özellikleri seçin
X = df[['Torsions', 'Gauss 1', 'Gauss 2', 'Pose Number', 'Hydrophobic', 'Repulsion', 'Hydrogen', 'Torsional']]
y = df['Affinity(kcal/mol)']

# Ligandlar için kontrol
train_ligands = df.loc[X.index, 'Ligand'].unique()

# Eğitim ve test setlerinde ortak ligandlar olmaması için veri ayırma
all_ligands = df['Ligand'].unique()
train_ligands, test_ligands = train_test_split(all_ligands, test_size=0.2, random_state=42)

# Eğitim ve test verilerini ayırmak için 'Ligand' sütununu kullanın
train_df = df[df['Ligand'].isin(train_ligands)]
test_df = df[df['Ligand'].isin(test_ligands)]

# Eğitim ve test verilerini ayırın
X_train = train_df[['Torsions', 'Gauss 1', 'Gauss 2', 'Pose Number', 'Hydrophobic', 'Repulsion', 'Hydrogen', 'Torsional']]
y_train = train_df['Affinity(kcal/mol)']
X_test = test_df[['Torsions', 'Gauss 1', 'Gauss 2', 'Pose Number', 'Hydrophobic', 'Repulsion', 'Hydrogen', 'Torsional']]
y_test = test_df['Affinity(kcal/mol)']

# Modeli tanımlayın ve eğitin
model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
model.fit(X_train, y_train)

# Test verileriyle tahmin yapın ve hata hesapla
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Modelin Ortalama Mutlak Hatası (MAE): {mae}")

# Ligand bazlı tahminleri ekleyelim
df['Predicted Affinity'] = model.predict(X)

# Çıkarılacak ligandlar
excluded_ligands = ['lig_90', 'lig_95', 'lig_22', 'lig_77', 'lig_39']

# Ligandları filtreleyelim
filtered_df = df[~df['Ligand'].isin(excluded_ligands)]

# Tekrar model eğitimi
X_filtered = filtered_df[['Torsions', 'Gauss 1', 'Gauss 2', 'Pose Number', 'Hydrophobic', 'Repulsion', 'Hydrogen', 'Torsional']]
y_filtered = filtered_df['Affinity(kcal/mol)']

# Eğitim ve test verilerini tekrar ayırın
train_ligands_filtered, test_ligands_filtered = train_test_split(filtered_df['Ligand'].unique(), test_size=0.2, random_state=42)

train_filtered_df = filtered_df[filtered_df['Ligand'].isin(train_ligands_filtered)]
test_filtered_df = filtered_df[filtered_df['Ligand'].isin(test_ligands_filtered)]

X_train_filtered = train_filtered_df[['Torsions', 'Gauss 1', 'Gauss 2', 'Pose Number', 'Hydrophobic', 'Repulsion', 'Hydrogen', 'Torsional']]
y_train_filtered = train_filtered_df['Affinity(kcal/mol)']
X_test_filtered = test_filtered_df[['Torsions', 'Gauss 1', 'Gauss 2', 'Pose Number', 'Hydrophobic', 'Repulsion', 'Hydrogen', 'Torsional']]
y_test_filtered = test_filtered_df['Affinity(kcal/mol)']

# Yeni model tahminleri ve doğruluk hesaplaması
model.fit(X_train_filtered, y_train_filtered)
y_pred_filtered = model.predict(X_test_filtered)
mae_filtered = mean_absolute_error(y_test_filtered, y_pred_filtered)
print(f"\nYeni Modelin Ortalama Mutlak Hatası (MAE): {mae_filtered}")

# En iyi 10 ligand ve tahmini afinitelerini terminale yazdır
top_10_ligands = filtered_df.drop_duplicates(subset=['Ligand']).nsmallest(10, 'Predicted Affinity')

print("\nTop 10 Predicted Ligands and Their Affinities:")
for index, row in top_10_ligands.iterrows():
    print(f"Ligand: {row['Ligand']}, Predicted Affinity: {row['Predicted Affinity']:.2f} kcal/mol")
