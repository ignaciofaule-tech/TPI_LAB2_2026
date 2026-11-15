import pandas as pd
import os
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt
import pickle
from Preprocesamiento.preprocessing_pipeline import preprocessing_pipeline
import argparse
import json
import subprocess
def train_model(path):
    params = {"n_estimators": 474, "max_depth": 2, "min_samples_split": 11, "min_samples_leaf": 24, "learning_rate": 0.02280463128210319}
    model = GradientBoostingRegressor(**params,random_state=42,loss='absolute_error')
    
    df = pd.read_csv(path)
    df = df.dropna(subset="target_Frio_kW_next")
    X = df.drop(columns=["target_Frio_kW_next"])
    y = df["target_Frio_kW_next"]
    X_preprocess = preprocessing_pipeline(X)
    model.fit(X_preprocess,y)
    
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Entrena el mejor modelo con los datos del path")
    
    parser.add_argument("--archivo",type=str,default="data/processed/dataset_final.csv",help="Path del .csv que se usará como train")
    parser.add_argument("--version",type=str,default="1.0.0",help="Versión del modelo en formato x.x.x")
    
    args = parser.parse_args()
    path = args.archivo
    version = args.version
    
    model = train_model(path)
    
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
                              "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")}

    with open(registry_path, "w") as f:
        json.dump(registry_data,f)
        