import pandas as pd
import numpy as np
import os


def procesar_archivo(filepath):
    xls = pd.ExcelFile(filepath)
    hojas_validas = [h for h in xls.sheet_names]

    df_consolidado = None

    for hoja in hojas_validas:
        df = pd.read_excel(filepath, sheet_name=hoja)
        df.columns = df.columns.astype(str).str.strip()

        cols_no_deseadas = [c for c in df.columns if 'Unnamed' in c or c == '' or c.lower() == 'id']
        df.drop(columns=cols_no_deseadas, inplace=True, errors='ignore')

        if 'DIA' not in df.columns or 'HORA' not in df.columns:
            continue

        df['DIA'] = pd.to_datetime(df['DIA'], errors='coerce').dt.date
        df['HORA'] = df['HORA'].astype(str).str.extract(r'(\d{1,2}:\d{2}:\d{2})')[0]
        df['HORA'] = pd.to_datetime(df['HORA'], format='%H:%M:%S', errors='coerce').dt.floor('min').dt.time

        df.dropna(subset=['DIA', 'HORA'], inplace=True)

        conteos = df.groupby('DIA').size()
        dias_completos = conteos[conteos.isin([24, 25])].index
        df_completos = df[df['DIA'].isin(dias_completos)]

        dias_con_error = df_completos.groupby('DIA').filter(lambda x: x['HORA'].duplicated().any())['DIA'].unique()
        for dia in dias_con_error:
            indices_error = df[df['DIA'] == dia].index
            cantidad = len(indices_error)

            horas_correctas = [pd.to_datetime(f"{h:02d}:00:00", format='%H:%M:%S').time() for h in range(24)]
            if cantidad == 25:
                horas_correctas.append(pd.to_datetime("23:59:00", format='%H:%M:%S').time())

            df.loc[indices_error, 'HORA'] = horas_correctas

        # Eliminar duplicados exactos evaluando ambas columnas simultáneamente
        df.drop_duplicates(subset=['DIA', 'HORA'], keep='last', inplace=True)
        df = df.loc[:, ~df.columns.duplicated()].copy()

        # Merge de las hojas
        if df_consolidado is None:
            df_consolidado = df
        else:
            df_consolidado = pd.merge(df_consolidado, df, on=['DIA', 'HORA'], how='outer', suffixes=('', '_dup'))
            cols_dup = [c for c in df_consolidado.columns if c.endswith('_dup')]
            df_consolidado.drop(columns=cols_dup, inplace=True)
            df_consolidado = df_consolidado.copy()

    return df_consolidado

def crear_df_diario(archivos):
    
    lista_dfs = []
    for archivo in archivos:
        if os.path.exists(archivo):
            print(f"Procesando: {archivo}...")
            df_archivo = procesar_archivo(archivo)
            lista_dfs.append(df_archivo)
        else:
            print(f"Archivo no encontrado (revisá la ruta): {archivo}")

    if len(lista_dfs) > 0:
        df_final = pd.concat(lista_dfs, ignore_index=True)
        df_final = df_final.copy()

        df_final.sort_values(by=['DIA', 'HORA'], inplace=True)
        duplicados_antes = len(df_final)
        df_final.drop_duplicates(subset=['DIA', 'HORA'], keep='last', inplace=True)
        duplicados_despues = len(df_final)

        print(f"\nSe eliminaron {duplicados_antes - duplicados_despues} registros solapados.")

        # Reordenar columnas
        columnas_ordenadas = ['DIA', 'HORA'] + [c for c in df_final.columns if c not in ['DIA', 'HORA']]
        df_final = df_final[columnas_ordenadas]
        df_final.reset_index(drop=True, inplace=True)

        print(f"Dimensión de la base horaria final: {df_final.shape}")


    df_temp = df_final.groupby('DIA').last().reset_index()
    hora_cierre = pd.to_datetime('23:59:00').time()
    df_diario = df_temp[df_temp['HORA'] == hora_cierre].copy()


    dias_eliminados = len(df_temp) - len(df_diario)
    print(f"Se eliminaron {dias_eliminados} días incompletos que no llegaron a las 23:59.")

    fecha_siguiente = df_diario["DIA"].shift(-1)
    fecha_actual = df_diario["DIA"]
    diferencia_dia_siguiente = fecha_siguiente - fecha_actual
    indices_a_borrar = df_diario.index[diferencia_dia_siguiente > pd.Timedelta(days=1)]
    df_diario = df_diario.drop(index=indices_a_borrar)
    df_diario['target_Frio_kW_next'] = df_diario['Frio (Kw)'].shift(-1)

    df_diario = df_diario.dropna(subset=['target_Frio_kW_next']).copy()

    #Features a utilizar (conocimiento del dominio)

    features_a_utilizar = ['DIA', 'HORA', 'EE Planta / Hl', 'Agua Planta / Hl', 'ET Planta / Hl',
       'Aire Planta / Hl', 'CO 2 / Hl', 'Hl de Mosto', 'Hl Producido Bodega',
       'Hl Cerveza Filtrada', 'Hl Cerveza Envasada','Elaboracion (Kw)', 'Bodega (Kw)', 'Envasado (Kw)',
       'Servicios (Kw)', 'Frio (Kw)', 'Agua Planta (Hl)', 'Agua Helada Cocina',
       'Glicol Paste L3', 'Temp Tq Intermedio', 'Gas Planta (Mj)',
       'Vapor Elaboracion (Kg)', 'Aire Producido (M3)',
       'Aire Planta (M3)', 'Totalizador_Bodega', 'Totalizador_Tq_Pulmon',
       'Tot Fermantacion_Cocina', 'Tot Reposo Inferior',
       'Tot  Reposo Superior', 'Tot  Trasiego', 'Tot A10/20',
       'Tot A40/240/50/60/Centec/Filtro', 'Tot  A130/330/430',
       'Tot L3, L4 y Planta de CO2', 'target_Frio_kW_next']

    df_diario = df_diario[features_a_utilizar].copy()
    
    return df_diario