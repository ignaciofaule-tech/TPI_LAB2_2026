import pandas as pd
import pickle
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from feature_engine.outliers import Winsorizer
from custom_transformers import OutlierHandler
import argparse
from sklearn.compose import ColumnTransformer

def create_preprocess_pipeline(datapath,train_size = 0.85, semilla=42,):

    df = pd.read_csv(datapath)

    target = 'target_Frio_kW_next'
    df = df.dropna(subset=["target_Frio_kW_next"]).copy()
    if train_size < 1:
        X_train, _, _, _ = train_test_split(df.drop(columns=[target]), df[target], train_size=train_size, shuffle=False, random_state=semilla)
    else:
        X_train = df.drop(columns=[target])
    columnas_a_escalar = X_train.columns[~X_train.columns.isin(["cos_month_of_year", "sin_month_of_year"])]
    
    Escalado = ColumnTransformer(
        transformers=[("Escalado", RobustScaler(), columnas_a_escalar)],
        remainder="passthrough",
        verbose_feature_names_out=False
    )


    preprocesamiento = Pipeline(steps=[
        ("OutlierHandler", OutlierHandler(fold=3)),
        ("Winsorizer", Winsorizer(capping_method='iqr', tail='right', fold=1.5,missing_values="ignore")),
        ("Escalado", Escalado),
        ("Imputacion", KNNImputer(n_neighbors=7))
    ]).set_output(transform="pandas")

    preprocesamiento.fit(X_train)

    return preprocesamiento


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Crea y entrena el pipeline de preprocesamiento.")


    parser.add_argument("--archivo",type=str,
                        default="data/processed/dataset_final.csv",
                        help='Nombre del archivos utilizado para entrenar el pipeline de preprocesamiento')
    parser.add_argument("--train_size", type=float,default=0.7,help="Tamaño del train (1 si no se quiere particionar)")
    parser.add_argument("--output", type=str,default="preprocesamiento_only_train",help="Nombre del archivo de salida (sin .pkl)")
    args = parser.parse_args()

    preprocesamiento = create_preprocess_pipeline(args.archivo,args.train_size)
    with open("src/Preprocesamiento/"+args.output+".pkl", "wb") as f:
        pickle.dump(preprocesamiento, f)
