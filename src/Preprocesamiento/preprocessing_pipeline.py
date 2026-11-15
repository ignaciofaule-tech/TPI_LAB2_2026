import pickle
import pandas as pd
import sys
sys.path.append("src/Preprocesamiento")
from custom_transformers import OutlierHandler

def preprocessing_pipeline(X,preprocess_path):
    with open(preprocess_path, "rb") as f:
        preprocesamiento = pickle.load(f)
    print("Preprocesando...")
    return preprocesamiento.transform(X)
if __name__ == "__main__":
    df_final = pd.read_csv("data/processed/dataset_final.csv")

    preprocessed_data = preprocessing_pipeline(df_final.drop(columns=["target_Frio_kW_next"]))
    preprocessed_data.to_csv("data/processed/dataset_final_preprocessed.csv", index=False)