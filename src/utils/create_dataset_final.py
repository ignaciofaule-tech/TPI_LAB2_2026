import numpy as np
import pandas as pd
import os
import argparse
from pathlib import Path

def create_dataset_final(filepath=None,df=None):
    if filepath is not None:
        datapath_completo = filepath
        df_completo = pd.read_csv(datapath_completo)
    elif df is not None:
        df_completo = df
    df_completo = df_completo.replace(' ', np.nan)
    columnas_numericas = ~df_completo.columns.isin(["DIA", "HORA"])
    df_completo.loc[:, columnas_numericas] = (
    df_completo.loc[:, columnas_numericas]
    .apply(pd.to_numeric, errors="coerce")
    )
    df_completo["HORA"] = df_completo["HORA"].astype(str)
    df_completo = df_completo.sort_values(["DIA", "HORA"])
    df_temp = df_completo.groupby('DIA').tail(1).reset_index(drop=True)
    hora_cierre = '23:59:00'
    df_diario = df_temp[df_temp['HORA'] == hora_cierre].copy()


    dias_eliminados = len(df_temp) - len(df_diario)
    print(f"Se eliminaron {dias_eliminados} días incompletos que no llegaron a las 23:59.")

    #días donde el target no se reinicio a las 00:00 (Se obtuvieron del eda)
    dias_sin_reinicio = df_completo[(df_completo["HORA"] == "00:00:00") & (df_completo["Frio (Kw)"] != 0)]["DIA"]
    df_diario = df_diario.drop(index= df_diario[df_diario["DIA"].isin(dias_sin_reinicio)].index)
    df_diario = df_diario[df_diario["Frio (Kw)"] <= 500000].copy()


    fecha_siguiente = pd.to_datetime(df_diario["DIA"].shift(-1))
    fecha_actual = pd.to_datetime(df_diario["DIA"])
    dias_consecutivos = (fecha_siguiente == fecha_actual + pd.Timedelta(days=1))
    df_diario["target_Frio_kW_next"] = (df_diario["Frio (Kw)"].shift(-1))
    df_diario.loc[~dias_consecutivos,"target_Frio_kW_next"] = np.nan
    #df_diario = df_diario.dropna(subset=["target_Frio_kW_next"]).copy()            

    mycoms = df_diario.columns[df_diario.columns.str.contains("Mycom")]
    df_diario["Kw Mycom Mean"] = df_diario.loc[:, mycoms].mean(axis=1, skipna=True)
    df_diario = df_diario.drop(columns=mycoms, errors= "ignore")

    df_diario["DIA"] = pd.to_datetime(df_diario["DIA"])
    df_diario["MES"] = df_diario["DIA"].dt.month

    fecha_pasada = pd.to_datetime(df_diario["DIA"].shift(7))
    fecha_actual = pd.to_datetime(df_diario["DIA"])
    diferencia_dia_pasado = fecha_actual - fecha_pasada

    df_diario['Frio lag last week'] = df_diario['Frio (Kw)'].shift(7)
    indices_a_borrar = df_diario.index[diferencia_dia_pasado != pd.Timedelta(days=7)]
    df_diario.loc[indices_a_borrar, 'Frio lag last week'] = np.nan 



    fecha_pasada = pd.to_datetime(df_diario["DIA"].shift(2))
    fecha_actual = pd.to_datetime(df_diario["DIA"])
    diferencia_dia_pasado = fecha_actual - fecha_pasada

    #lag plazo corto
    df_diario['Frio lag 2 days'] = df_diario['Frio (Kw)'].shift(2)
    indices_a_borrar = df_diario.index[diferencia_dia_pasado != pd.Timedelta(days=2)]
    df_diario.loc[indices_a_borrar, 'Frio lag 2 days'] = np.nan 


    df_diario["Prom movil Frio"] = df_diario["Frio (Kw)"].rolling(window=7).mean()

    df_diario['sin_month_of_year'] = np.sin(2 * np.pi * (df_diario["MES"] ) / 12)
    df_diario['cos_month_of_year'] = np.cos(2 * np.pi * (df_diario["MES"] ) / 12)

    #poly y ratios
    df_diario['Ratio_Frio_Planta'] = (df_diario['Frio (Kw)'] / (df_diario['KW Gral Planta'] + 1e-5))
    df_diario["Ratio_Cond_Frio"] = df_diario["KW Cond 5. 6 y 9"] / (df_diario["Frio (Kw)"] + 1e-5)


    #poly y ratios
    df_diario['Ratio_Frio_Planta'] = (df_diario['Frio (Kw)'] / (df_diario['KW Gral Planta'] + 1e-5))
    df_diario["Ratio_Cond_Frio"] = df_diario["KW Cond 5. 6 y 9"] / (df_diario["Frio (Kw)"] + 1e-5)


    
    #Features a utilizar (conocimiento del dominio)


    fechas = df_diario["DIA"].values
    

    features_a_utilizar = [
        'Prom movil Frio',
        'Frio (Kw)',
        'Servicios (Kw)',
        'Sala Maq (Kw)',
        'EE Frio / Hl',
        'Bodega (Kw)',
        'cos_month_of_year',
        'sin_month_of_year',
        'Frio lag last week',
        'Kw Mycom Mean',
        'KW Gral Planta',
        'Agua Cond REC',
        'Ratio_Frio_Planta',
        'Hl Cerveza Filtrada',
        'Envasado (Kw)',
        'KW Cond 5. 6 y 9',
        'Ratio_Cond_Frio',
        'Frio lag 2 days'
    ]

    df_completo["HORA"] = df_completo["HORA"].astype(str)


    target = ["target_Frio_kW_next"]
    df_diario = df_diario[features_a_utilizar + target].copy()

    return df_diario.copy(), fechas


if __name__ == "__main__":
    from create_checksum import create_checksum
    parser = argparse.ArgumentParser(description="Crea el dataset final.")

    parser.add_argument("--archivo",type=str,
                        default="data/clean/dataset_consolidado_complete.csv",
                        help='Nombre del archivos utilizado para crear el df final')

    parser.add_argument("--output",type=str,default="dataset_final",help="Nombre del nuevo archivo Limpio (no añadir .csv)")


    args = parser.parse_args()
    archivo = args.archivo
    output = args.output
    df_final,_ = create_dataset_final(archivo)
    path = os.path.join("data","processed",output+".csv")

    df_final.to_csv(path, index=False)

    create_checksum(json_path= Path(r"data\checksum.json"), datos_path=Path(path))



