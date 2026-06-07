import pandas as pd
import glob
import os

def procesar_a_csv(carpeta_base, colector, tipo_feature, nombre_salida):
    print(f"Buscando {tipo_feature} en la carpeta {carpeta_base}...")
    datos = []
    
    # Traverse the directory structure targeting 'normal' (0) and 'ataque' (1) subsets
    for label_str in ['normal', 'ataque']:
        # Path pattern: base_folder/label/event_name/transform/feature_type/*.json
        patron = os.path.join(carpeta_base, label_str, "*", "transform", tipo_feature, "*.json")
        archivos = glob.glob(patron)
        
        for f in archivos:
            try:
                # Normalize path to handle cross-OS slashes and extract the event name
                partes_ruta = os.path.normpath(f).split(os.sep)
                evento = partes_ruta[2]  # e.g., 'YouTube_Hijack'
                
                df = pd.read_json(f)
                
                # Transpose dataframe if BML exported features horizontally
                if df.shape[0] < df.shape[1]: 
                    df = df.T
                
                # Inject essential metadata for downstream Machine Learning tasks
                df['Evento'] = evento
                df['Label'] = 0 if label_str == 'normal' else 1
                df['Colector'] = colector
                datos.append(df)
            except Exception as e:
                print(f"  ⚠️ Error en archivo {f}: {e}")

    # Data consolidation and export
    if datos:
        df_final = pd.concat(datos).fillna(0)
        df_final.index.name = 'Timestamp'
        df_final.reset_index(inplace=True)
        df_final.to_csv(nombre_salida, index=False)
        print(f"ÉXITO: {nombre_salida} creado con {len(df_final)} registros.\n")
    else:
        print(f"FALLO: No se encontraron datos en {carpeta_base} para {tipo_feature}.\n")

# ==============================================================================
# PIPELINE EXECUTION: GENERATING THE 4 CORE DATASETS
# ==============================================================================

procesar_a_csv("dataset_rrc04", "rrc04", "Features", "volumen_rrc04.csv")
procesar_a_csv("dataset_rrc04", "rrc04", "GraphFeatures", "grafos_rrc04.csv")

procesar_a_csv("dataset_rrc06", "rrc06", "Features", "volumen_rrc06.csv")
procesar_a_csv("dataset_rrc06", "rrc06", "GraphFeatures", "grafos_rrc06.csv")