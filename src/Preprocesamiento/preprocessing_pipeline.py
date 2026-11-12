import pickle
import pandas as pd

def preprocessing_pipeline(X):
    with open("src/Preprocesamiento/preprocesamiento.pkl", "rb") as f:
        preprocesamiento = pickle.load(f)
    return preprocesamiento.transform(X)

df_final = pd.read_csv("data/processed/dataset_final.csv")

preprocessed_data = preprocessing_pipeline(df_final.drop(columns=["target_Frio_kW_next"]))
preprocessed_data.to_csv("data/processed/dataset_final_preprocessed.csv", index=False)