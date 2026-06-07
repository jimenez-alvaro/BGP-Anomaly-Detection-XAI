import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import shap
import matplotlib.pyplot as plt
import re

plt.rcParams['font.family'] = 'sans-serif'

# ==============================================================================
# 1. DATA LOADING AND METHODOLOGICAL PREPARATION
# ==============================================================================
print("Cargando dataset y aplicando aislamiento Z-Score...")
df = pd.read_csv('dataset_csvs/grafos_rrc04.csv')

# Semantic incident matching
df['Incidente_Base'] = df['Evento'].apply(
    lambda x: re.sub(r'_(Normal|Hijack|Leak|Outage)$', '', x)
)
# Remove corrupted data
df = df[df['Incidente_Base'] != 'Swisscom']

cols_X = df.drop(
    columns=['Timestamp', 'Evento', 'Incidente_Base', 'Label', 'Colector'],
    errors='ignore'
).columns

# Event-Isolated Z-Score Normalization (prevents data leakage)
for base in df['Incidente_Base'].unique():
    idx_all    = df['Incidente_Base'] == base
    idx_normal = (df['Incidente_Base'] == base) & (df['Label'] == 0)
    
    scaler = StandardScaler()
    # Fit exclusively on baseline normal traffic
    scaler.fit(df.loc[idx_normal, cols_X])
    df.loc[idx_all, cols_X] = scaler.transform(df.loc[idx_all, cols_X])

X = df[cols_X]
y = df['Label']

# Data splitting consistent with the evaluation pipeline
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================================================================
# 2. XGBOOST MODEL TRAINING
# ==============================================================================
print("Entrenando modelo XGBoost...")
modelo_xgb = xgb.XGBClassifier(eval_metric='logloss', random_state=42, n_jobs=-1)
modelo_xgb.fit(X_train, y_train)

# ==============================================================================
# 3. EXPLAINABLE AI (SHAP) ANALYSIS
# ==============================================================================
print("Calculando valores SHAP...")
explainer   = shap.TreeExplainer(modelo_xgb)
shap_values = explainer.shap_values(X_test)

# ==============================================================================
# 4. PUBLICATION-QUALITY FIGURE GENERATION
# ==============================================================================

# --- Bar plot: global feature importance ---
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values, X_test,
    plot_type="bar",
    show=False
)
ax = plt.gca()
ax.set_title("Global Feature Importance (XGBoost)", fontsize=13)
ax.set_xlabel("mean(|SHAP value|) (average impact on model output magnitude)", fontsize=11)
ax.set_ylabel("Feature", fontsize=11)
plt.tight_layout()
plt.savefig("shap_importancia_barras.pdf", dpi=300, bbox_inches='tight')
print("✅ Bar plot saved as 'shap_importancia_barras.pdf'")
plt.close()

# --- Dot plot: per-sample feature impact ---
plt.figure(figsize=(10, 6))
shap.summary_plot(
    shap_values, X_test,
    show=False
)
ax = plt.gca()
ax.set_title("Feature Impact on Anomaly Predictions (XGBoost)", fontsize=13)
ax.set_xlabel("SHAP value (impact on model output)", fontsize=11)
ax.set_ylabel("Feature", fontsize=11)

# Adjust colorbar labels for the dot plot
try:
    cbar = plt.gcf().axes[-1]
    cbar.set_ylabel("Feature value", fontsize=10)
    cbar.set_yticks([0, 1])
    cbar.set_yticklabels(["Low", "High"], fontsize=9)
except Exception:
    pass

plt.tight_layout()
plt.savefig("shap_impacto_puntos.pdf", dpi=300, bbox_inches='tight')
print("✅ Dot plot saved as 'shap_impacto_puntos.pdf'")
plt.close()