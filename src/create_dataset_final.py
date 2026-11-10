import numpy as np
import pandas as pd
import os
import argparse

def crear_dataset_completo(archivo, output):

    FILENAME = archivo
    datapath_completo = os.path.join("data","clean",FILENAME)
    df_completo = pd.read_csv(datapath_completo)

    df_completo = df_completo.replace(' ', np.nan)
    columnas_numericas = ~df_completo.columns.isin(["DIA", "HORA"])
    df_completo.loc[:, columnas_numericas] = (
    df_completo.loc[:, columnas_numericas]
    .apply(pd.to_numeric, errors="coerce")
    )

    df_completo = df_completo.sort_values(["DIA", "HORA"])
    df_temp = df_completo.groupby('DIA').tail(1).reset_index(drop=True)
    hora_cierre = '23:59:00'
    df_diario = df_temp[df_temp['HORA'] == hora_cierre].copy()


    dias_eliminados = len(df_temp) - len(df_diario)
    print(f"Se eliminaron {dias_eliminados} días incompletos que no llegaron a las 23:59.")

    #días donde el target no se reinicio a las 00:00 (Se obtuvieron del eda)
    dias_sin_reinicio = df_completo[(df_completo["HORA"] == "00:00:00") & (df_completo["Frio (Kw)"] != 0)]["DIA"]
    df_diario = df_diario.drop(index= df_diario[df_diario["DIA"].isin(dias_sin_reinicio)].index)

    fecha_siguiente = pd.to_datetime(df_diario["DIA"].shift(-1))
    fecha_actual = pd.to_datetime(df_diario["DIA"])
    dias_consecutivos = (fecha_siguiente == fecha_actual + pd.Timedelta(days=1))
    df_diario["target_Frio_kW_next"] = (df_diario["Frio (Kw)"].shift(-1))
    df_diario.loc[~dias_consecutivos,"target_Frio_kW_next"] = np.nan
    df_diario = df_diario.dropna(subset=["target_Frio_kW_next"]).copy()            

    mycoms = df_diario.columns[df_diario.columns.str.contains("Mycom")]
    df_diario["Kw Mycom Mean"] = df_diario.loc[:, mycoms].mean(axis=1, skipna=True)
    df_diario = df_diario.drop(columns=mycoms, errors= "ignore")

    df_diario["DIA"] = pd.to_datetime(df_diario["DIA"])
    
    df_diario["MES"] = df_diario["DIA"].dt.month

    fecha_pasada = pd.to_datetime(df_diario["DIA"].shift(7))
    fecha_actual = pd.to_datetime(df_diario["DIA"])
    diferencia_dia_pasado = fecha_actual - fecha_pasada

    df_diario['Frio lag last week'] = df_diario['Frio (Kw)'].shift(7)
    indices_a_borrar = df_diario.index[diferencia_dia_pasado > pd.Timedelta(days=7)]
    df_diario.loc[indices_a_borrar, 'Frio lag last week'] = None 

    df_diario["Prom movil Frio"] = df_diario["Frio (Kw)"].rolling(window=7).mean()

    
    df_diario['sin_month_of_year'] = np.sin(2 * np.pi * df_diario["MES"] / 12)
    df_diario['cos_month_of_year'] = np.cos(2 * np.pi * df_diario["MES"] / 12)

    
    #Features a utilizar (conocimiento del dominio)

    features_a_utilizar = [
        "Kw Mycom Mean",
        'Frio (Kw)', 
        'cos_month_of_year',
        'Prom movil Frio',
        'KW Gral Planta',
        'KW Cond 5. 6 y 9',
        'EE Frio / Hl',
        'Temp Tq Intermedio',
        'Frio lag last week',
        'Sala Maq (Kw)',
        'Kw Mycom Mean',
        'Hl Cerveza Filtrada',
        'Servicios (Kw)',
        'Tot  Trasiego',
        'Agua Cond REC',
        'Vapor Elaboracion (Kg)',
        'sin_month_of_year'
    ]

    df_completo["HORA"] = df_completo["HORA"].astype(str)


    target = ["target_Frio_kW_next"]
    df_diario = df_diario[features_a_utilizar + target].copy()

    # Superan millon target
    idx_millones = df_diario[df_diario["target_Frio_kW_next"] > 500000].index
    df_diario.drop(index=idx_millones)

    df_diario.to_csv(os.path.join("data","processed",output+".csv"))




parser = argparse.ArgumentParser(description="Crea el dataset final.")

parser.add_argument("--archivo",type=str,
                    default="dataset_consolidado_complete.csv",
                    help='Nombre del archivos utilizado para crear el df final')

parser.add_argument("--output",type=str,default="dataset_final",help="Nombre del nuevo archivo Limpio (no añadir .csv)")


args = parser.parse_args()
archivo = args.archivo
output = args.output
crear_dataset_completo(archivo,output)