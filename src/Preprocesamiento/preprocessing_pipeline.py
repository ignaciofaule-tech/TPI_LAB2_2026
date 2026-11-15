import pickle
import pandas as pd
import sys
sys.path.append("src/Preprocesamiento")
from custom_transformers import OutlierHandler

def preprocessing_pipeline(X):
    with open("src/Preprocesamiento/preprocesamiento.pkl", "rb") as f:
        preprocesamiento = pickle.load(f)
    return preprocesamiento.transform(X)
if __name__ == "__main__":
    df_final = pd.read_csv("data/processed/dataset_final.csv")

    preprocessed_data = preprocessing_pipeline(df_final.drop(columns=["target_Frio_kW_next"]))
    preprocessed_data.to_csv("data/processed/dataset_final_preprocessed.csv", index=False)