import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def load_kdd_dataset(data_path):
    """Lectura del dataSet NSL-KDD."""
    import arff  # Importación específica necesaria para esta función
    with open(data_path, 'r') as train_set:
        dataset = arff.load(train_set)
    attributes = [attr[0] for attr in dataset["attributes"]]
    return pd.DataFrame(dataset["data"], columns=attributes)

# Ruta al archivo CSV
def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'datasets', 'obesidad.csv')
    df = pd.read_csv(csv_path)

    # Información general del dataset
    df.info()

    # Separar el dataset de datos: 60% train_set, 40% test_set
    train_set, test_set = train_test_split(df, test_size=0.4, random_state=42)

    # Información de los conjuntos de datos
    train_set.info()
    test_set.info()

    # Separar el dataset de pruebas: 50% validation_set, 50% test_set
    val_set, test_set = train_test_split(test_set, test_size=0.5, random_state=42)

    print("Longitud del training_set:", len(train_set))
    print("logitud de validation_set:", len(val_set))
    print("longitud del test_set:", len(test_set))

    # Método de particionado aleatorio con shuffle=False
    train_set, test_set = train_test_split(df, test_size=0.4, random_state=42, shuffle=False)

    # Método de particionado estratificado
    train_set, test_set = train_test_split(df, test_size=0.4, random_state=42, stratify=df["Gender"])

    # Función de particionado completo
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

    print("longitud del set de datos", len(df))

    train_set, val_set, test_set = train_val_test_split(df, stratify='Gender')

    print("Longitud del training_set:", len(train_set))
    print("logitud de validation_set:", len(val_set))
    print("longitud del test_set:", len(test_set))

    # Comprobar que stratify mantiene la proporción de las características en los conjuntos
    df["Gender"].hist()
    plt.title("Distribución de 'Gender' en el dataset completo")
    plt.show()

    train_set["Gender"].hist()
    plt.title("Distribución de 'Gender' en el training set")
    plt.show()

    val_set["Gender"].hist()
    plt.title("Distribución de 'Gender' en el validation set")
    plt.show()

    test_set["Gender"].hist()
    plt.title("Distribución de 'Gender' en el test set")
    plt.show()

if __name__ == "__main__":
    main()
