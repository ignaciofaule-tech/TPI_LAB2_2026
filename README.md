# TPI_LAB2_2026: Predicción de Consumo Energético Industrial

**Objetivo:** Desarrollar un pipeline de machine learning completo y reproducible para predecir el consumo energético
del sistema de refrigeración Frio (kW) del día siguiente en una planta cervecera mexicana.

## Entorno de desarrollo
- **Python:** 3.12
- **Gestor:** conda

## Pasos de Instalación
1) Clonar repositorio
2) Crear entorno: `conda create --name cervecera_env python=3.12`
3) Activar entorno: `conda activate cervecera_env`
4) Instalar dependencias: `pip install -r requirements.txt`


## Estructura del proyecto
- `data/`: Datos crudos, limpios y procesados.
- `src/`: Código fuente para la creación del dataset, el preprocesamiento, modelado y predicción.
- `notebooks/`: Notebooks jupyter de EDA, preprocesamiento y modelado.  
- `results/`: Resultados de los experimentos realizados con disitntos modelos.
- `models/`: Modelo entrenado y un registro de versionado del mismo.


## Ejecución del Entrenamiento 

**ACLARACIÓN:** Este paso es puramente opcional. Si desea re entrenar el modelo con nuevos datos, sigue estos pasos:

1) Abre tu terminal
2) Asegúrate de tener activado el entorno virtual (ejecutando `conda activate cervecera_env`).
3) Cargar el dataset (.xlsx) de tus datos de entrenamiento a la carpeta del proyecto. Se recomienda ubicarla en la carpeta de data para mantener la organización.
4) Verifica que estás posicionado en la carpeta principal del proyecto (`TPI_LAB2_2026`).
5) Ejecuta el script del preprocesamiento (fundamental para una buena transformación de los datos) ingresando este comando y presionando Enter: 

   `python src/create_preprocess_pipeline.py --archivo [path_del_archivo] --train_size 0.85 --output [nombre_pipeline]`

    train_size es el tamaño del train. En caso de valer 1, entrena con todo el dataset.

6) Ejecuta el script de entrenamiento ingresando este comando y presionando Enter: 

   `python src/train_model.py --archivo [path_del_archivo] --train_size 0.85 --preprocess_path [path_preprocess_pipeline]`

**OPCIONAL:** el archivo de predict tiene adicionalmente un argento de versión de modelo. Si se desea utilizar una versión que no sea la predeterminada, **ejecute el paso 5** de la siguiente forma: 

`python src/train_model.py --archivo [path_del_archivo] --version [version_deseada] --train_size 0.85 --preprocess_path [path_preprocess_pipeline]` 


## Ejecución de la predicción

Para obtener la predicción del consumo energético dado un dataset, se debe realizar los siguientes pasos: 

1) Abre tu terminal.
2) Asegúrate de tener activado el entorno virtual (ejecutando `conda activate cervecera_env`).
4) Cargar el dataset (.xlsx) a la carpeta del proyecto. Se recomienda ubicarla en la carpeta de data para mantener la organización.
3) Posicionarse (si es que no lo esta) en la carpeta principal del proyecto (`TPI_LAB2_2026`).
4) Ejecuta el script de predicción (modificando para que coincida con el path de sus datos) ingresando este comando y presionando Enter:
   `python src/predict.pyc --archivo [path_del_archivo]`

**OPCIONAL:** el archivo de predict tiene adicionalmente un argento de versión de modelo. Si se desea utilizar una versión que no sea la actual (puede fijarse la versiones existentes), **ejecute el paso 4** de la siguiente forma: `python src/predict.pyc --archivo [path_del_archivo] --version [version_deseada]`


## Aclaraciones:

Los archivos crudos de los totalizadores son ignorados y la obtención de los mismo fue mediante la plataforma virtual de la materia. 