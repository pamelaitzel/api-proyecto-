#%matplotlib inline
import os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline

# Construccion de una funcion que 
def train_val_test_split(df, rstate = 42, shuffle = True, stratify = None):
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size = 0.4, random_state =rstate, shuffle=shuffle, stratify=strat)
    strat = test_set[stratify] if stratify else  None
    val_set, test_set = train_test_split(
        test_set, test_size = 0.5, random_state = rstate, shuffle=shuffle, stratify=strat)
    return (train_set, val_set, test_set)

# Representacion grafica del limite de desición.
def plot_svm_decision_boundary(svm_clf, xmin, xmax):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]

    #At the decision boundary, w0*x0 + w1*x1 + b =0
    # => z1 = -w/w1 * x0 -b/w1
    x0 = np-linspace(xmin, xmax, 200)
    decision_boundary = -w[0]/w[1] * x0 -b/w[1]

    margin = 1/w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin

    svm = cvm_clf.support_vectors_
    plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors = '#FFAAAA')
    plt.plot(x0, decision_boundary, "k-", kinewidth=2)
    plt.plot(x0, gutter_up, "k--", linewidth=2)
    plt.plot(x0, gutter_down, "k--", linewidth=2)

base_dir = os.path.dirname(os.path.abspath(__file__))

# Ruta al archivo CSV
csv_path = os.path.join(base_dir, 'datasets', 'obesidad.csv')

# Cargar el archivo CSV en un DataFrame
df = pd.read_csv(csv_path)

# 2.- Visualizacion de la informacion
df.head(10)

df.describe()

df.info()

df["Gender"].value_counts()

is_null = df.isna().any()
is_null[is_null]

# Comprobar si exiten valores infinitos.
is_inf = df.isin([np.inf, -np.inf]).any()
is_inf[is_inf]

# Representacion grafica de 2 caracteristicas.
plt.figure(figsize=(12, 6))

# Hombres
plt.scatter(df["Weight"][df['Gender'] == "Male"], 
            df["Height"][df['Gender'] == "Male"], 
            c="r", marker=".", label="Male")

# Mujeres
plt.scatter(df["Weight"][df['Gender'] == "Female"], 
            df["Height"][df['Gender'] == "Female"], 
            c="g", marker="x", label="Female")

# Etiquetas y leyenda
plt.xlabel("Weight", fontsize=13)
plt.ylabel("Height", fontsize=13)
plt.legend()
plt.show()

train_set, val_set, test_set = train_val_test_split(df)

train_set = pd.DataFrame(train_set) 

X_train = train_set.drop("Gender", axis=1)
y_train = train_set["Gender"].copy()

X_val = val_set.drop("Gender", axis=1)
y_val = val_set["Gender"].copy()

X_test = test_set.drop("Gender", axis=1)
y_test = test_set["Gender"].copy()

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")

import pandas as pd
from sklearn.preprocessing import StandardScaler

# Verifica los tipos de datos
print(X_train.dtypes)

# Convertir variables categóricas a numéricas
X_train = pd.get_dummies(X_train, drop_first=True)
X_val = pd.get_dummies(X_val, drop_first=True)
X_test = pd.get_dummies(X_test, drop_first=True)

# Asegurar consistencia entre conjuntos
X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# Escalado
scaler = StandardScaler()
X_train_prep = scaler.fit_transform(X_train)
X_val_prep = scaler.transform(X_val)
X_test_prep = scaler.transform(X_test)

# Convertir de nuevo a DataFrame (opcional)
X_train_prep = pd.DataFrame(X_train_prep, columns=X_train.columns, index=X_train.index)
X_val_prep = pd.DataFrame(X_val_prep, columns=X_val.columns, index=X_val.index)
X_test_prep = pd.DataFrame(X_test_prep, columns=X_test.columns, index=X_test.index)

print(X_train_prep.head())

X_train_prep.head(10)

# Comprobar si exiten valores nulos en dataset de entrenamiento 
is_null = X_train_prep.isna().any()
is_null[is_null]

X_train_reduced = X_train_prep[["Weight","Height"]]
X_val_reduced = X_val_prep[["Weight","Height"].copy()]
X_train_reduced

from sklearn.svm import SVC 

# SVM Large Margin Clasification

svm_clf = SVC(kernel="linear", C=50)
svm_clf.fit(X_train_reduced, y_train)
def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]

    #At the decision boundary, w0*x0 + w1*x1 + b =0
    # => z1 = -w/w1 * x0 -b/w1
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0]/w[1] * x0 -b/w[1]

    margin = 1/w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin

    svs = svm_clf.support_vectors_
    plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors = '#FFAAAA')
    plt.plot(x0, decision_boundary, "k-", linewidth=2)
    plt.plot(x0, gutter_up, "k--", linewidth=2)
    plt.plot(x0, gutter_down, "k--", linewidth=2)
    plt.figure(figsize = (12,6))
plt.plot(X_train_reduced.values[:, 0] [y_train =="Male"], X_train_reduced.values[:, 1][y_train=="Male"], "g^")
plt.plot(X_train_reduced.values[:, 0] [y_train =="Female"], X_train_reduced.values[:, 1][y_train=="Female"], "bs")
plot_svc_decision_boundary(svm_clf, 0, 1)
plt.title("$C = {}$".format(svm_clf.C), fontsize=18)
plt.axis([0, 1,  -100, 200])
plt.xlabel("Weight", fontsize=13)
plt.ylabel("Height", fontsize=13)
plt.show()

y_pred = svm_clf.predict(X_val_reduced)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Preprocesamiento (asegúrate de que los datos sean numéricos y escalados)
scaler = StandardScaler()
X_train_prep = scaler.fit_transform(X_train)
X_val_prep = scaler.transform(X_val)

# Entrenamiento del modelo
model = LogisticRegression()
model.fit(X_train_prep, y_train)

# Predicciones
y_pred = model.predict(X_val_prep)

# Cálculo del F1 Score
print("F1 Score:", f1_score(y_val, y_pred, pos_label='Male'))

svm_clf_sc = Pipeline([
    ("scaler", RobustScaler()),
    ("linear_svc", SVC(kernel="linear", C=50))
    ])

svm_clf_sc.fit(X_train_reduced, y_train)

y_pred = svm_clf_sc.predict(X_val_reduced)
print("F1 Score:", f1_score(y_pred, y_val, pos_label='Male'))

# entrenamiento con todo el dataset
from sklearn.svm import SVC

svm_clf = SVC(kernel="linear", C=1)
svm_clf.fit(X_train_prep, y_train)
y_pred = svm_clf.predict(X_val_prep)
print("F1 Score", f1_score(y_pred, y_val, pos_label='Male'))

# para representar el limite de decision, se tiene que pasar 
#la variable objetivo  a numerica
y_train_num = y_train.factorize()[0]
y_val_num = y_val.factorize()[0]
from sklearn.datasets import make_moons
from sklearn.svm import LinearSVC
from sklearn.preprocessing import PolynomialFeatures
polynomial_svm_clf = Pipeline([
    ("poly_features", PolynomialFeatures(degree=3)),
    ("scaler", StandardScaler()),
    ("svm_clf", LinearSVC(C=20, loss="hinge", random_state=42, max_iter=100000))
])

polynomial_svm_clf.fit(X_train_reduced, y_train_num)

def plot_dataset(X, y):
    plt.plot(X[:,0][y==1], X[:,1] [y==1],"g.")
    plt.plot(X[:,0][y==0], X[:,1] [y==0],"b.")

    # Importar la función necesaria
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt

# Generar un conjunto de datos de ejemplo
X, y = make_classification(
    n_features=2, n_redundant=0, n_clusters_per_class=1, random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Escalar los datos
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar un modelo SVM
polynomial_svm_clf = SVC(kernel="poly", degree=3, coef0=1, C=5)
polynomial_svm_clf.fit(X_train_scaled, y_train)

# Función para graficar los datos
def plot_dataset(X, y, xlabel="Feature 1", ylabel="Feature 2"):
    """
    Graficar un dataset con dos características y sus etiquetas.
    """
    plt.scatter(X[:, 0][y == 0], X[:, 1][y == 0], color="red", label="Clase 0", marker="o")
    plt.scatter(X[:, 0][y == 1], X[:, 1][y == 1], color="blue", label="Clase 1", marker="s")
    plt.xlabel(xlabel, fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.legend()
    plt.grid()

# Función para graficar las predicciones de un modelo
def plot_predictions(clf, axes):
    """
    Graficar las predicciones de un clasificador en el espacio de características.
    """
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X_new = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X_new).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, alpha=0.3, cmap="coolwarm")
    plt.contour(x0, x1, y_pred, colors="k", levels=[0.5], alpha=0.5)

# Graficar los datos y las predicciones
fig, axes = plt.subplots(ncols=2, figsize=(15, 5), sharey=True)

# Primer gráfico: Datos de entrenamiento
plt.sca(axes[0])
plot_dataset(X_train_scaled, y_train, xlabel="Feature 1 (scaled)", ylabel="Feature 2 (scaled)")

# Segundo gráfico: Predicciones del modelo
plt.sca(axes[1])
plot_predictions(polynomial_svm_clf, [-3, 3, -3, 3])  # Ajusta los límites según tus datos
plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, cmap="coolwarm", edgecolors="k")

plt.show()

y_pred = polynomial_svm_clf.predict(X_val_reduced)
print("F1 Score: ", f1_score(y_pred, y_val_num))

svm_clf = SVC(kernel="poly", degree=3, coef0=10, C=20)
svm_clf.fit(X_train_reduced, y_train_num)
fig, axes = plt.subplots(ncols=2, figsize=(15,5), sharey=True)
plt.sca(axes[0])
plot_dataset(X_train_reduced.values, y_train_num)
plot_predictions(svm_clf, [0, 1, -100, 250])
plt.xlabel("Weight", fontsize=11)
plt.ylabel("Weight", fontsize=11)
plt.sca(axes[1])
plot_predictions(svm_clf, [0, 1, -100, 250])
plt.xlabel("Weight", fontsize=11)
plt.ylabel("Weight", fontsize=11)
plt.show()

y_pred = polynomial_svm_clf.predict(X_val_reduced)
print('F1 Score: ', f1_score(y_pred, y_val_num))
rbf_kernel_svm_clf = Pipeline([
    ("scaler", RobustScaler()),
    ("svm_clf", SVC(kernel="rbf", gamma=0.05, C=1000))
])
rbf_kernel_svm_clf.fit(X_train_prep, y_train_num)
y_pred = rbf_kernel_svm_clf.predict(X_val_prep)
print('F1 Score: ', f1_score(y_pred, y_val_num))