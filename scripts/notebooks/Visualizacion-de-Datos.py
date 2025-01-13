import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from pandas.plotting import scatter_matrix

# 1. Cargar el dataset
base_dir = os.path.dirname(os.path.abspath(__file__))

# Ruta al archivo CSV
csv_path = os.path.join(base_dir, 'datasets', 'obesidad.csv')

# Cargar el archivo CSV en un DataFrame
df = pd.read_csv(csv_path)


# 2. Inspección básica
print("Primeras filas del dataset:")
print(df.head())
print("\nInformación general del dataset:")
df.info()
print("\nEstadísticas descriptivas:")
print(df.describe())

# 3. Valores únicos de columnas seleccionadas
if "Age" in df.columns:
    print("\nValores únicos en 'Age':")
    print(df["Age"].value_counts())
else:
    print("\nLa columna 'Age' no existe en el dataset.")

if "Weight" in df.columns:
    print("\nValores únicos en 'Weight':")
    print(df["Weight"].value_counts())
else:
    print("\nLa columna 'Weight' no existe en el dataset.")

# 4. Histograma de la columna "Weight"
if "Weight" in df.columns:
    df["Weight"].hist()
    plt.title("Distribución de 'Weight'")
    plt.xlabel("Peso")
    plt.ylabel("Frecuencia")
    plt.show()
else:
    print("\nNo se puede graficar 'Weight' porque no existe en el dataset.")

# 5. Histograma general
df.hist(bins=50, figsize=(20, 15))
plt.suptitle("Distribución de los atributos")
plt.show()

# 6. Transformación de valores categóricos
if "Weight" in df.columns:
    labelencoder = LabelEncoder()
    df["Weight"] = labelencoder.fit_transform(df["Weight"])
    print("\nTransformación de 'Weight' a valores numéricos realizada.")
else:
    print("\nNo se pudo transformar 'Weight' porque no existe en el dataset.")

# 7. Matriz de correlación
corr_matrix = df.corr(numeric_only=True)
if not corr_matrix.empty:
    print("\nCorrelación con 'Weight':")
    if "Weight" in corr_matrix.columns:
        print(corr_matrix["Weight"].sort_values(ascending=False))
    else:
        print("'Weight' no está presente en la matriz de correlación.")
else:
    print("\nNo se pudo calcular la matriz de correlación.")

# 8. Visualización de la matriz de correlación
plt.figure(figsize=(8, 8))
plt.matshow(corr_matrix, fignum=1)
plt.title("Matriz de correlación", pad=20)
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.colorbar()
plt.show()

# 9. Scatter matrix
attributes = ["Age", "Height", "Weight", "FCVC"]
valid_attributes = [attr for attr in attributes if attr in df.columns]

if valid_attributes:
    scatter_matrix(df[valid_attributes], figsize=(12, 8))
    plt.suptitle("Scatter Matrix de atributos seleccionados")
    plt.show()
else:
    print("\nNo hay atributos válidos para crear la Scatter Matrix.")
