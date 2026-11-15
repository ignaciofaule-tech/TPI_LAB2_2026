import pandas as pd
import os
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import pickle
from Preprocesamiento.preprocessing_pipeline import preprocessing_pipeline
from sklearn.model_selection import train_test_split
import argparse
import json
import subprocess
def train_model(path,preprocess_path,train_size=0.85,semilla=42):
    params = {"n_estimators": 474, "max_depth": 2, "min_samples_split": 11, "min_samples_leaf": 24, "learning_rate": 0.02280463128210319}
    model = GradientBoostingRegressor(**params,random_state=42,loss='absolute_error')
    
    df = pd.read_csv(path)
    df = df.dropna(subset="target_Frio_kW_next")
    X = df.drop(columns=["target_Frio_kW_next"])
    y = df["target_Frio_kW_next"]
    if train_size < 1:
        X_train,_,y_train,_ = train_test_split(X,y,train_size=train_size,shuffle=False,random_state=semilla)
    else:
        X_train = X
        y_train = y
    X_preprocess = preprocessing_pipeline(X_train,preprocess_path)
    model.fit(X_preprocess,y_train)
    
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Entrena el mejor modelo con los datos del path")
    
    parser.add_argument("--archivo",type=str,default="data/processed/dataset_final.csv",help="Path del .csv que se usará como train")
    parser.add_argument("--train_size",type=float, default = 0.85,help='Tamaño del train (1 si no se quiere particionar)')
    parser.add_argument("--version",type=str,default="1.0.0",help="Versión del modelo en formato x.x.x")
    parser.add_argument("--preprocess_path",type=str,default="src/Preprocesamiento/preprocesamiento_only_train.pkl",help="path del pipeline de preprocesamiento")
    
    args = parser.parse_args()
    path = args.archivo
    version = args.version
    train_size = args.train_size
    preprocess_path = args.preprocess_path
    model = train_model(path,preprocess_path,train_size)
    
    registry_path = "models/model_registry.json"
    registry_data = {}

    model_path = "models/modelo_v" + version + ".pkl" 
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    if os.path.exists(registry_path):
        with open(registry_path,"r") as f:
            registry_data = json.load(f)
    
    registry_data[version] = {"version": version,
                              "path": model_path,
                              "hiperparametros": model.get_params(),
                              "preprocess_path": preprocess_path,
                              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")}

    with open(registry_path, "w") as f:
        json.dump(registry_data,f)
        