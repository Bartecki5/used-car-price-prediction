# 🚗 Used Car Price Prediction

Ten projekt to kompletny rurociąg (pipeline) Machine Learning, którego celem jest precyzyjne przewidywanie cen samochodów używanych na podstawie ich parametrów technicznych i historii. 

## 🛠️ Etapy projektu
1. **Czyszczenie Danych (Data Cleaning):** Usuwanie zbędnych i zaszumionych kolumn, obsługa braków danych (NaN) przy użyciu mody grupowanej po konkretnych modelach aut.
2. **Inżynieria Cech (Feature Engineering):** Obliczanie faktycznego wieku pojazdu w czasie rzeczywistym na podstawie roku produkcji oraz usuwanie wartości odstających (Outliers).
3. **Analiza Danych (EDA):** Badanie korelacji oraz wizualizacja rozkładu cen i cech aut (wykresy punktowe, pudełkowe, macierz korelacji).
4. **Kodowanie Zmiennych (Encoding):** Przekształcanie kategorycznych danych tekstowych na liczbowe za pomocą `LabelEncoder` oraz `One-Hot Encoding`.
5. **Modelowanie i Ewaluacja:** Trenowanie dwóch potężnych modeli uczenia maszynowego i zderzenie ich wyników.

## 🏆 Wyniki (Skuteczność Modeli)
W projekcie przetestowano i porównano dwa algorytmy. Ostateczne wyniki na odseparowanym zbiorze testowym prezentują się następująco:

* **Random Forest Regressor:**
  * Dopasowanie ($R^2$): **0.9030**
  * Średni Błąd (MAE): **~54,057**

* **XGBoost Regressor (Zwycięzca 🥇):**
  * Dopasowanie ($R^2$): **0.9083**
  * Średni Błąd (MAE): **~51,056**

Oba algorytmy poradziły sobie znakomicie (ponad 90% wyjaśnionej wariancji), jednak w tym zestawieniu **XGBoost** okazał się nieznacznie skuteczniejszy, popełniając mniejszy średni błąd przy wycenie.

## 💻 Wykorzystane Technologie
* **Język:** Python
* **Przetwarzanie danych:** Pandas, NumPy
* **Wizualizacja:** Matplotlib, Seaborn
* **Machine Learning:** Scikit-Learn, XGBoost
