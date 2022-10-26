import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dane = pd.read_csv("zadanie.csv", sep=',')

"""Zadanie 1"""
col = dane.columns;  # wczytanie nazw kolumn

val = dane.values;  # wczytanie wartosci tablicy

mean_col = val.mean(axis=0)  # srednia dla kazdej kolumny
# mean_col = np.mean(dane, axis=(0)) - II sposob (wynik typu Series)

mean_std = np.std(val)  # odchylenie st. dla calej tablicy

difference = val - mean_std  # roznica miedzy kazda wartoscia a odchyleniem

max_row_val = val.max(axis=1)  # max w wierszach
# max_row_val = np.max(dane, axis=(1)) - II sposob (wynik typu Series)

arr2 = 2 * val  # podwojona tablica wartosci

# nazwa kolumny z wartoscia maksymalna
max_value = val.max()
col_max = val.max(axis=0)
data_col = np.array(col)
mask = data_col[col_max == max_value]
print(mask)
mask_boolean = (col_max == max_value)

arr9 = (val<mean_std).sum(axis=0)  # liczba elementow spelniajacych warunek w kazdej kolumnie

"""Zadanie 2"""
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

X = dane.iloc[:, :dane.shape[1]-1]
y = dane.iloc[:, -1]

# Wykresy korelacji miedzy cechami niezaleznymi a zalezna
fig, ax = plt.subplots(X.shape[1], 1, figsize=(5, 10))
for i, column in enumerate(X.columns):
    ax[i].scatter(X[column], y)
    
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=221, shuffle=False)

""" Tworzenie oraz uczenie modelu regresji liniowej """
linReg = LinearRegression()  # twoezenie obiektu typu LinearRegression
linReg.fit(X_train, y_train)  # uczenie przekazujac podzbior uczacy
y_pred = linReg.predict(X_test)  # testowanie przekazujac podzbior testowy

# Generowanie wykresu
minval = min(y_test.min(), y_pred.min())
maxval = max(y_test.max(), y_pred.max())
plt.scatter(y_test, y_pred)
plt.plot([minval, maxval], [minval, maxval])
plt.xlabel('y_test')
plt.ylabel('y_pred')

mae = mean_absolute_error(y_test, y_pred)
print(mae)
