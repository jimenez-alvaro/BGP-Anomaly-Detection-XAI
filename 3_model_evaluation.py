import pandas as pd
import numpy as np
import re, os, time
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score, accuracy_score, f1_score
import xgboost as xgb

# ==============================================================================
# SETUP
# ==============================================================================
warnings.filterwarnings("ignore", category=FutureWarning)

# Define dataset paths and target files
ruta_csvs = "dataset_csvs/"
archivos = ["volumen_rrc04.csv", "grafos_rrc04.csv", "volumen_rrc06.csv", "grafos_rrc06.csv"]

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def get_category(name):
    """
    Semantic event classifier.
    Groups the 20 events into 3 macro-categories based on their filenames 
    (e.g., 'Facebook_Outage' -> 'Outage') to evaluate per-category performance.
    """
    if "Hijack" in name: return "Origin/Path Hijack"
    if "Leak" in name: return "Route Leak"
    if "Outage" in name: return "Outage"
    return "Other"

# ==============================================================================
# MODEL INITIALIZATION
# ==============================================================================

modelos = {
    "Logistic_Reg": LogisticRegression(max_iter=1000, random_state=42),
    "Random_Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    "XGBoost": xgb.XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1),
    "MLP_NeuralNet": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True, random_state=42)
}

resultados_resumen = []

# ==============================================================================
# MAIN LOOP: TRAINING AND EVALUATION
# ==============================================================================

for archivo in archivos:
    print(f"\n--- PROCESANDO DATASET: {archivo} ---")
    
    df = pd.read_csv(os.path.join(ruta_csvs, archivo))
    
    # --- DATA ENGINEERING: YEAR EFFECT MITIGATION ---
    
    # 1. Semantic Matching
    df['Incidente_Base'] = df['Evento'].apply(lambda x: re.sub(r'_(Normal|Hijack|Leak|Outage)$', '', x))
    
    # Remove 'Swisscom' event to prevent Z-Score calculation errors
    df = df[df['Incidente_Base'] != 'Swisscom']
    
    df['Categoria'] = df['Evento'].apply(get_category)
    cols_X = df.drop(columns=['Timestamp', 'Evento', 'Incidente_Base', 'Label', 'Colector', 'Categoria']).columns
    
    # 2. Event-Isolated Z-Score Normalization
    for base in df['Incidente_Base'].unique():
        # Mask for the entire event (normal + attack windows)
        idx_all = df['Incidente_Base'] == base
        
        # Mask for baseline traffic only (Label == 0)
        idx_normal = (df['Incidente_Base'] == base) & (df['Label'] == 0)
        
        scaler = StandardScaler()
        
        # Fit exclusively on baseline data to learn normal traffic distribution
        scaler.fit(df.loc[idx_normal, cols_X])
        
        # Transform the entire event vector to expose the anomaly without data leakage
        df.loc[idx_all, cols_X] = scaler.transform(df.loc[idx_all, cols_X])
    
    X = df[cols_X]
    y = df['Label']
    
    # --- EVALUATION: STRATIFIED K-FOLD CROSS-VALIDATION ---
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for nombre_mod, modelo in modelos.items():
        y_true, y_pred, cats = [], [], []
        
        for train_idx, test_idx in skf.split(X, y):
            modelo.fit(X.iloc[train_idx], y.iloc[train_idx])
            p = modelo.predict(X.iloc[test_idx])
            
            y_true.extend(y.iloc[test_idx])
            y_pred.extend(p)
            cats.extend(df['Categoria'].iloc[test_idx])

        # --- GLOBAL METRICS EXTRACTION ---
        acc = accuracy_score(y_true, y_pred)
        rec_global = recall_score(y_true, y_pred)
        f1_global = f1_score(y_true, y_pred)
        
        res_tmp = {
            "Dataset": archivo, 
            "Modelo": nombre_mod, 
            "Acc_Global": acc, 
            "F1_Global": f1_global, 
            "Rec_Global": rec_global
        }
        
        # --- PER-CATEGORY METRICS EXTRACTION ---
        for c in ["Origin/Path Hijack", "Route Leak", "Outage"]:
            mask = np.array(cats) == c
            y_t_sub = np.array(y_true)[mask]
            y_p_sub = np.array(y_pred)[mask]
            
            res_tmp[f"Rec_{c}"] = recall_score(y_t_sub, y_p_sub) if len(y_t_sub) > 0 else 0
            
        resultados_resumen.append(res_tmp)

# ==============================================================================
# FINAL EXPORT
# ==============================================================================

df_res = pd.DataFrame(resultados_resumen)

print("\n" + "="*100 + "\n RESULTADOS COMPARATIVOS FINALES\n" + "="*100)
print(df_res.to_string(index=False))

df_res.to_csv("resultados_comparativa_tfg.csv", index=False)