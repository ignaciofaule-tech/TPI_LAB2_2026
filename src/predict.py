import pandas as pd
import joblib
import sys
import json
import numpy as np
from utils.load_data import load_data
from utils.create_dataset_final import create_dataset_final
from Preprocesamiento.preprocessing_pipeline import preprocessing_pipeline
import os
import argparse
def load_model(version='latest'):
    """Carga el modelo y los preprocesadores desde el registro.
    'latest' carga el modelo con la version mas alta.
    """
    registry_path = 'models/model_registry.json'
    if not os.path.exists(registry_path):
        raise FileNotFoundError(f"No se encontró el archivo {registry_path}")
    with open(registry_path, 'r') as f:
        registry = json.load(f)
    # Logica para encontrar la ultima/mejor version
    if version=='latest':
        version = max(registry.keys(), key = lambda v: [int(x) for x in v.split(".")])
    if version not in registry:
        raise KeyError(f"La versión {version} no existe en el registro")
    model_path = registry[version]['path']
    preprocess_path = registry[version]['preprocess_path']
    model = joblib.load(model_path)
    print(f"Usando model_v{version}.pkl")
    return model, preprocess_path

def load_and_preprocess(filepath,preprocess_path):
    """Carga el archivo Excel y aplica todo el preprocesamiento.
    Reusa el pipeline de preprocessing_pipeline.py: no lo reimplementa.
    Devuelve la matriz de caracteristicas y las fechas correspondientes.
    """
    # ... aplicar pipeline de preprocesamiento ...
    df = load_data([filepath])
    df_final, fechas = create_dataset_final(df=df)
    target = df_final["target_Frio_kW_next"].values
    X_processed = preprocessing_pipeline(df_final.drop(columns=["target_Frio_kW_next"]),preprocess_path)
    return X_processed, fechas, target

def predict_consumption(filepath,version='latest'):
    """Funcion principal de prediccion."""
    model,preprocess_path = load_model(version) # carga desde el registry
    X, fechas, target = load_and_preprocess(filepath,preprocess_path)
    predictions = model.predict(X)
    # La prediccion es DIARIA: una fila por dia, no por hora.
    # 'fecha' es el dia D; la prediccion corresponde al consumo del dia D+1.
    return pd.DataFrame({
        'fecha': fechas,
        'prediccion_frio_kw': predictions,
        'target': target})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Realiza la predicción sobre un conjunto de datos")
    parser.add_argument("--archivo",type=str,help='Path del archivo xlsx')
    parser.add_argument("--version",type=str,default='latest',help="Version del modelo")
    
    args = parser.parse_args()
    
    results = predict_consumption(args.archivo,args.version)
    results.to_csv('results/predicciones.csv', index=False)
    print("Predicciones generadas en results/predicciones.csv")
