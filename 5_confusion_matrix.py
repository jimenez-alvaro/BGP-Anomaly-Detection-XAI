import pandas as pd
import numpy as np
import re, os
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# ==============================================================================
# INITIAL CONFIGURATION
# ==============================================================================
ruta_csvs = "dataset_csvs/"
archivos = ["volumen_rrc04.csv", "grafos_rrc04.csv"]
nombres_plot = ["Volumen", "Grafos"]

# ==============================================================================
# MAIN LOOP: CONFUSION MATRIX GENERATION (VOLUMETRIC AND TOPOLOGICAL)
# ==============================================================================

for archivo, nombre_plot in zip(archivos, nombres_plot):
    print(f"\n--- GENERANDO MATRIZ DE CONFUSIÓN PARA: {nombre_plot} ({archivo}) ---")
    
    # 1. DATA LOADING
    df = pd.read_csv(os.path.join(ruta_csvs, archivo))
    
    # 2. DATA ENGINEERING (Event-Isolated Z-Score Normalization)
    df['Incidente_Base'] = df['Evento'].apply(lambda x: re.sub(r'_(Normal|Hijack|Leak|Outage)$', '', x))
    df = df[df['Incidente_Base'] != 'Swisscom'] # Remove corrupted data
    
    cols_X = df.drop(columns=['Timestamp', 'Evento', 'Incidente_Base', 'Label', 'Colector', 'Categoria'], errors='ignore').columns
    
    for base in df['Incidente_Base'].unique():
        idx_all = df['Incidente_Base'] == base
        idx_normal = (df['Incidente_Base'] == base) & (df['Label'] == 0)
        
        scaler = StandardScaler()
        # Fit exclusively on baseline data to prevent data leakage
        scaler.fit(df.loc[idx_normal, cols_X])
        df.loc[idx_all, cols_X] = scaler.transform(df.loc[idx_all, cols_X])
    
    X = df[cols_X]
    y = df['Label']
    
    # 3. DATA SPLITTING (Consistent with SHAP analysis partition)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. XGBOOST TRAINING
    print("Entrenando modelo XGBoost...")
    modelo_xgb = xgb.XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1)
    modelo_xgb.fit(X_train, y_train)
    
    # 5. PREDICTION
    y_pred = modelo_xgb.predict(X_test)
    
    # 6. CONFUSION MATRIX CALCULATION
    cm = confusion_matrix(y_test, y_pred)
    
    # 7. PLOT GENERATION (Heatmap)
    plt.figure(figsize=(7, 5))
    
    # Class labels (0 = Normal, 1 = Anomaly)
    etiquetas = ['Normal (0)', 'Anomalía (1)']
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=etiquetas, yticklabels=etiquetas, annot_kws={"size": 14})
    
    plt.title(f'Matriz de Confusión XGBoost - Paradigm: {nombre_plot}', fontsize=14, pad=15)
    plt.xlabel('Predicción del Modelo', fontsize=12)
    plt.ylabel('Valor Real (Ground Truth)', fontsize=12)
    
    # Save as PDF for high-quality LaTeX rendering
    nombre_archivo_pdf = f"matriz_confusion_{nombre_plot.lower()}_rrc04.pdf"
    plt.tight_layout()
    plt.savefig(nombre_archivo_pdf, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Matriz de confusión guardada como '{nombre_archivo_pdf}'")
    
print("\nProceso finalizado. Puedes encontrar los dos PDFs en tu directorio.")