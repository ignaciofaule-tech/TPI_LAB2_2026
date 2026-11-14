import pandas as pd
import joblib
import sys
import json
import numpy as np
from utils import load_data, create_dataset_final
from Preprocesamiento import preprocessing_pipeline
import os
def load_model(version='latest'):
    """Carga el modelo y los preprocesadores desde el registro.
    'latest' carga el modelo con la version mas alta.
    """
    registry_path = 'models/model_registry.json'
    if not os.path.exist(registry_path):
        raise FileNotFoundError(f"No se encontró el archivo {registry_path}")

    with open(registry_path, 'r') as f:
        registry = json.load(f)
    # Logica para encontrar la ultima/mejor version
    if version=='latest':
        version = max(registry.keys(), key = lambda v: [int(x) for x in v.split(".")])
    if version not in registry:
        raise KeyError(f"La versión {version} no existe en el registro")
    model_path = registry[version]['path']
    model = joblib.load(model_path)
    return model

def load_and_preprocess(filepath):
    """Carga el archivo Excel y aplica todo el preprocesamiento.
    Reusa el pipeline de preprocessing_pipeline.py: no lo reimplementa.
    Devuelve la matriz de caracteristicas y las fechas correspondientes.
    """
    # ... aplicar pipeline de preprocesamiento ...
    df = load_data(filepath)
    df_final, fechas = create_dataset_final(df=df)
    X_processed = preprocessing_pipeline(df_final)
    return X_processed, fechas

def predict_consumption(filepath):
    """Funcion principal de prediccion."""
    model = load_model() # carga desde el registry
    X, fechas = load_and_preprocess(filepath)
    predictions = model.predict(X)
    # La prediccion es DIARIA: una fila por dia, no por hora.
    # 'fecha' es el dia D; la prediccion corresponde al consumo del dia D+1.
    return pd.DataFrame({
        'fecha': fechas,
        'prediccion_frio_kw': predictions})

if __name__ == "__main__":
    results = predict_consumption(sys.argv[1])
    results.to_csv('results/predicciones.csv', index=False)
    print("Predicciones generadas en results/predicciones.csv")
