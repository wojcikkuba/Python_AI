import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.svm import SVC as SVM
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier as kNN
from sklearn.pipeline import Pipeline

"""Zad1"""

data = pd.read_excel("zadanie_1.xlsx")

#zamiana na 0 i 1
def qualitative_to_0_1(data, column, value_to_be_1):
    columns = list(data.columns)
    mask = data[column].values == value_to_be_1
    data[column][mask] = 1
    data[column][~mask] = 0
    return data

data = qualitative_to_0_1(data, 'Gender', 'Female')
data = qualitative_to_0_1(data, 'Married', 'Yes')
data = qualitative_to_0_1(data, 'Education', 'Graduate')
data = qualitative_to_0_1(data, 'Self_Employed', 'Yes')
data = qualitative_to_0_1(data, 'Loan_Status', 'Y')

cat_feature = pd.Categorical(data.Property_Area)
one_hot = pd.get_dummies(cat_feature)
data = pd.concat([data, one_hot], axis=1)
data = data.drop(columns=['Property_Area'])

#podzial na podzbiory
features = data.columns
vals = data.values.astype(np.float)
y = data['Loan_Status'].astype(np.float)
X = data.drop(columns = ['Loan_Status']).values.astype(np.float)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

#StandardScaler i kNN
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

models = [kNN(n_neighbors=5, weights='distance'), SVM(kernel='sigmoid')]
for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(confusion_matrix(y_test, y_pred))
    
    
"""Zad2"""
data = pd.read_csv('zadanie_2.csv', sep=',')

def qualitative_to_0_1(data, column, value_to_be_1):
    columns = list(data.columns)
    mask = data[column].values == value_to_be_1
    data[column][mask] = 1
    data[column][~mask] = 0
    return data

data = qualitative_to_0_1(data, 'label', 'female')

#podzial na podzbiory
features = list(data.columns)
vals = data.values.astype(np.float)
X = vals[:,:-1]
y = vals[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#wykres
X_paced = PCA(2).fit_transform(X_train) # 2 glowne skladowe
fig, ax = plt.subplots(1, 1)
females = y_train == 1
ax.scatter(X_paced[females,0], X_paced[females,1], label='female')
ax.scatter(X_paced[~females,0], X_paced[~females,1], label='male')
ax.legend()

# nalezy uruchomic ponizszy skrypt przed powyzszym aby uzyskac inny wykres
scaler = StandardScaler() # #zastosowanie scalera
X_train = scaler.fit_transform(X_train) 
pca_transform = PCA()
pca_transform.fit(X_train)
variances = pca_transform.explained_variance_ratio_ 
cumulated_variances = variances.cumsum()
plt.scatter(np.arange(variances.shape[0]), cumulated_variances)
plt.yticks(np.arange(0, 1.1, 0.1))
PC_num = (cumulated_variances<0.95).sum()

#MinMaxScaler
scaler_MinMax = MinMaxScaler()
scaler_MinMax.fit(X_train)
X_train = scaler_MinMax.transform(X_train)
X_test = scaler_MinMax.transform(X_test)

#klasyfikacja
pipe = Pipeline([['transformer', PCA(7)],
                 ['scaler', MinMaxScaler()],
                ['classifier',kNN(n_neighbors=5, weights='distance')]])
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(confusion_matrix(y_test, y_pred))
