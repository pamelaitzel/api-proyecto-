import os
import arff
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, f1_score, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt


# ## funciones auxiliares
def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat)
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat)
    return train_set, val_set, test_set


# Construcción de un pipeline para los atributos numéricos
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('rbst_scaler', RobustScaler())
])


# Transformador para codificar columnas categóricas
class CustomOneHotEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._oh = OneHotEncoder(sparse_output=False)
        self._columns = None

    def fit(self, X, y=None):
        X_cat = X.select_dtypes(include=["object"])
        self._columns = pd.get_dummies(X_cat).columns
        self._oh.fit(X_cat)
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        X_cat = X_copy.select_dtypes(include=["object"])
        X_num = X_copy.select_dtypes(exclude=["object"])
        X_cat_oh = self._oh.transform(X_cat)
        X_cat_oh = pd.DataFrame(X_cat_oh, columns=self._columns, index=X_copy.index)
        X_copy.drop(list(X_cat), axis=1, inplace=True)
        return X_copy.join(X_cat_oh)


class DataFramePreparer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self._full_pipeline = None
        self._columns = None

    def fit(self, X, y=None):
        num_attribs = list(X.select_dtypes(exclude=["object"]))
        cat_attribs = list(X.select_dtypes(include=["object"]))
        self._full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attribs),
            ("cat", CustomOneHotEncoder(), cat_attribs),
        ])
        self._full_pipeline.fit(X)
        self._columns = pd.get_dummies(X).columns
        return self

    def transform(self, X, y=None):
        X_copy = X.copy()
        X_prep = self._full_pipeline.transform(X_copy)
        return pd.DataFrame(X_prep, columns=self._columns, index=X_copy.index)


# Lectura del dataset
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, 'datasets', 'obesidad.csv')
df = pd.read_csv(csv_path)

# División del DataSet
train_set, val_set, test_set = train_val_test_split(df)

# Separación de características y etiquetas
X_train = train_set.drop("Gender", axis=1)
y_train = train_set["Gender"].copy()

X_val = val_set.drop("Gender", axis=1)
y_val = val_set["Gender"].copy()

X_test = test_set.drop("Gender", axis=1)
y_test = test_set["Gender"].copy()

# Preparación del DataSet
data_preparer = DataFramePreparer()
data_preparer.fit(df.drop("Gender", axis=1))

X_train_prep = data_preparer.transform(X_train)
X_val_prep = data_preparer.transform(X_val)
X_test_prep = data_preparer.transform(X_test)

# Entrenamiento de un clasificador SVM con kernel RBF
rbf_kernel_svm_clf = SVC(kernel='rbf', probability=True)
rbf_kernel_svm_clf.fit(X_train_prep, y_train)

# Predicción con el subconjunto de prueba
y_pred = rbf_kernel_svm_clf.predict(X_test_prep)

# Métricas y visualización
ConfusionMatrixDisplay.from_estimator(rbf_kernel_svm_clf, X_test_prep, y_test, values_format='d')
plt.title("Confusion Matrix")
plt.show()

print("F1 score (micro):", f1_score(y_test, y_pred, average="micro"))
print("F1 score (macro):", f1_score(y_test, y_pred, average="macro"))
print("F1 score (weighted):", f1_score(y_test, y_pred, average="weighted"))

# Curvas ROC y PR
RocCurveDisplay.from_estimator(rbf_kernel_svm_clf, X_test_prep, y_test)
plt.title("ROC Curve")
plt.show()

PrecisionRecallDisplay.from_estimator(rbf_kernel_svm_clf, X_test_prep, y_test)
plt.title("Precision-Recall Curve")
plt.show()



