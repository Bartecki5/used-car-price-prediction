import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor


# 1. ŁADOWANIE DANYCH
print("--- 1. ŁADOWANIE DANYCH ---")
df = pd.read_csv("Used_Car_Price_Prediction.csv")
wszystkie_wiersze_start = len(df)
print(f"Początkowa liczba aut: {wszystkie_wiersze_start}")


# 2. CZYSZCZENIE DANYCH I USUWANIE BRAKÓW
print("\n--- 2. CZYSZCZENIE DANYCH ---")
kolumny_do_usuniecia = [
    'car_name', 'city', 'registered_city', 'registered_state', 
    'times_viewed', 'is_hot', 'source', 'ad_created_on', 'reserved', 
    'rto', 'car_availability', 'broker_quote', 'emi_starts_from', 
    'booking_down_pymnt', 'original_price' # original_price usunięte z powodu dużej liczby braków
]
df = df.drop(columns=kolumny_do_usuniecia, errors='ignore')

# Usuwanie wierszy, gdzie brakuje kluczowych i trudnych do zastąpienia danych
df = df.dropna(subset=['car_rating', 'fitness_certificate'])

# Funkcja do inteligentnego uzupełniania braków na podstawie modelu auta
def napraw_braki_w_grupie(grupa_aut):
    moda = grupa_aut.mode()
    if len(moda) > 0:
        najczestsza_wartosc = moda.iloc[0]
        return grupa_aut.fillna(najczestsza_wartosc)
    return grupa_aut

df['transmission'] = df.groupby('model')['transmission'].transform(napraw_braki_w_grupie)
df['body_type'] = df.groupby('model')['body_type'].transform(napraw_braki_w_grupie)

# Usunięcie niedobitków, jeśli dla jakiegoś rzadkiego modelu w ogóle nie było mody
df = df.dropna(subset=['body_type', 'transmission'])


# 3. FILTROWANIE OUTLIERÓW (WYJĄTKÓW)
df = df[df["kms_run"] < 400000]
df = df[df["yr_mfr"] > 2000]
df = df[df["sale_price"] < 3500000]


# 4. INŻYNIERIA CECH (FEATURE ENGINEERING)
obecny_rok = datetime.datetime.now().year
df["car_age"] = obecny_rok - df["yr_mfr"]
df = df.drop(columns=["yr_mfr"])

print(f"Liczba aut gotowych do analizy: {len(df)}")


# 5. EXPLORATORY DATA ANALYSIS (EDA)
print("\n--- 3. GENEROWANIE WYKRESÓW (ZAMKNIJ OKNA, ABY TRENOWAĆ MODELE) ---")

# Wykresy punktowe dla cech liczbowych
cechy_liczbowe = ["kms_run", "car_age", "total_owners"]
plt.figure(figsize=(15, 5))
for i, col in enumerate(cechy_liczbowe, 1):
    plt.subplot(1, 3, i)
    sns.scatterplot(data=df, x=col, y="sale_price", alpha=0.6)
    plt.title(f"{col} vs Cena")
    plt.grid(True)
plt.tight_layout()
plt.show()

# Wykresy pudełkowe dla cech kategorycznych
cechy_kategoryczne = ['fuel_type', 'transmission', "body_type"]
for col in cechy_kategoryczne:
    if col in df.columns:
        plt.figure(figsize=(10, 5))
        sns.boxplot(data=df, x=col, y='sale_price')
        plt.title(f"Rozkład cen dla: {col}")
        plt.grid(axis="y")
        plt.show()

# Heatmapa korelacji
plt.figure(figsize=(8, 6))
kolumny_liczbowe = ['sale_price', "kms_run", "car_age", "total_owners"]
corr_matrix = df[kolumny_liczbowe].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Macierz korelacji")
plt.show()


# 6. KODOWANIE CECH (PRZYGOTOWANIE DLA MODELI)
print("\n--- 4. TRENOWANIE MODELI ---")
le_make = LabelEncoder()
le_model = LabelEncoder()
le_variant = LabelEncoder()
le_condition = LabelEncoder()

df['make_code'] = le_make.fit_transform(df['make']) 
df['model_code'] = le_model.fit_transform(df['model'])
df['variant_code'] = le_variant.fit_transform(df['variant'])
df['condition_code'] = le_condition.fit_transform(df['car_rating'])

df = df.drop(columns=["make", "model", "variant", "car_rating"], errors='ignore')

# One-Hot Encoding
df = pd.get_dummies(df, columns=["fuel_type", "transmission", "body_type", "assured_buy", "fitness_certificate", "warranty_avail"], dtype=int)


# 7. TRENOWANIE I OCENA MODELI
X = df.drop('sale_price', axis=1)
y = df['sale_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- RANDOM FOREST ---
rf_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

r2_rf = r2_score(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)

print("-" * 40)
print(f"RANDOM FOREST:")
print(f"Dopasowanie (R2): {r2_rf:.4f}")
print(f"Średni Błąd (MAE): {mae_rf:,.0f} PLN/USD")
print("-" * 40)

# --- XGBOOST ---
xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=-1, random_state=42)
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)

r2_xgb = r2_score(y_test, y_pred_xgb)
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)

print(f"XGBOOST:")
print(f"Dopasowanie (R2): {r2_xgb:.4f}")
print(f"Średni Błąd (MAE): {mae_xgb:,.0f} PLN/USD")
print("-" * 40)


# 8. PODSUMOWANIE WYNIKÓW
print("\n--- WNIOSKI KOŃCOWE ---")
if r2_xgb > r2_rf:
    print("Zwycięzca: XGBoost")
else:
    print("Zwycięzca: Random")