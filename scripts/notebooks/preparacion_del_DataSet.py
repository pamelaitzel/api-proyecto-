import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat
    )
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat
    )
    return train_set, val_set, test_set

base_dir = os.path.dirname(os.path.abspath(__file__))

# Ruta al archivo CSV
csv_path = os.path.join(base_dir, 'datasets', 'obesidad.csv')

# Cargar el archivo CSV en un DataFrame
df = pd.read_csv(csv_path)

train_set, val_set, test_set = train_val_test_split(df, stratify='Gender')

print("Longitud del training_set:", len(train_set))
print("logitud de validation_set:", len(val_set))
print("longitud del test_set:", len(test_set))

# Separar las características de entrada de las características de salida
X_train = train_set.drop("Weight", axis=1)
y_train = train_set["Weight"].copy()

print(X_train.columns)

# Para ilustrar esta sección, añadimos valores nulos a una característica existente
X_train.loc[(X_train["Height"] > 60) & (X_train["Height"] < 200), "Height"] = np.nan

# Comprobar si existen atributos con valores nulos
print("Atributos con valores nulos:", X_train.isna().any())

# Seleccionar las filas que contengan valores nulos
filas_valores_nulos = X_train[X_train.isnull().any(axis=1)]
print("Filas con valores nulos:")
print(filas_valores_nulos)

# Copiar el dataset para no alterar el original
X_train_copy = X_train.copy()

# Rellenar los valores nulos de "Height" con la mediana
mediana_height = X_train_copy["Height"].median()
X_train_copy.fillna({"Height": mediana_height}, inplace=True)

# Imputar valores nulos usando SimpleImputer
imputer = SimpleImputer(strategy="median")

# Seleccionar solo los atributos numéricos
X_train_copy_num = X_train_copy.select_dtypes(exclude=["object"])

# Ajustar el imputer y transformar los datos
imputer.fit(X_train_copy_num)
X_train_copy_num_nonam = imputer.transform(X_train_copy_num)

# Crear un nuevo DataFrame con los datos imputados
X_train_copy_imputed = pd.DataFrame(X_train_copy_num_nonam, columns=X_train_copy_num.columns)

print("Primeras 10 filas después de la imputación:")
print(X_train_copy_imputed.head(10))

