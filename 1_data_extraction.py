from BML.data import Dataset
from BML.transform import DatasetTransformation
from BML import utils

print("Iniciando extracción BML (Volumen y Grafos separados - 40 eventos | 3 colectores)")

# ==============================================================================
# PIPELINE CONFIGURATION
# ==============================================================================

folder = "dataset_final/"
dataset = Dataset(folder)

# Global extraction parameters: 2-hour graph priming period to prevent cold starts
dataset.setParams({
    "PrimingPeriod": 2 * 60 * 60,
    "IpVersion": [4],
    "Collectors": ["rrc04"] 
})

def crear_evento(nombre, label, y, m, d, h_start, h_end):
    return {
        "name": nombre, "label": label,
        "start_time": utils.getTimestamp(y, m, d, h_start, 0, 0),
        "end_time": utils.getTimestamp(y, m, d, h_end, 0, 0)
    }

# ==============================================================================
# EVENT DEFINITION (ATTACK vs BASELINE PARADIGM)
# ==============================================================================

dataset.setPeriodsOfInterests([
    # --- 1. ORIGIN HIJACKS ---
    crear_evento("YouTube_Normal", "normal", 2008, 2, 23, 18, 21),
    crear_evento("YouTube_Hijack", "ataque", 2008, 2, 24, 18, 21),
    crear_evento("Amazon_MEW_Normal", "normal", 2018, 4, 23, 11, 14),
    crear_evento("Amazon_MEW_Hijack", "ataque", 2018, 4, 24, 11, 14),
    crear_evento("Rostelecom_Normal", "normal", 2020, 3, 31, 19, 22),
    crear_evento("Rostelecom_Hijack", "ataque", 2020, 4, 1, 19, 22),
    crear_evento("Twitter_Normal", "normal", 2022, 2, 27, 10, 13),
    crear_evento("Twitter_Hijack", "ataque", 2022, 2, 28, 10, 13),
    crear_evento("CelerNetwork_Normal", "normal", 2022, 8, 16, 19, 22),
    crear_evento("CelerNetwork_Hijack", "ataque", 2022, 8, 17, 19, 22),

    # --- 2. PATH HIJACKS ---
    crear_evento("Bitcoin_AS32653_Normal", "normal", 2014, 8, 6, 14, 17),
    crear_evento("Bitcoin_AS32653_Hijack", "ataque", 2014, 8, 7, 14, 17),
    crear_evento("Visa_Rostelecom_Normal", "normal", 2017, 12, 11, 10, 13),
    crear_evento("Visa_Rostelecom_Hijack", "ataque", 2017, 12, 12, 10, 13),
    crear_evento("Bitcanal_Normal", "normal", 2018, 7, 8, 10, 13),
    crear_evento("Bitcanal_Hijack", "ataque", 2018, 7, 9, 10, 13),
    crear_evento("ChinaTelecom_Normal", "normal", 2019, 6, 5, 10, 13),
    crear_evento("ChinaTelecom_Hijack", "ataque", 2019, 6, 6, 10, 13),
    crear_evento("Swisscom_Normal", "normal", 2019, 12, 8, 9, 12),
    crear_evento("Swisscom_Hijack", "ataque", 2019, 12, 9, 9, 12),

    # --- 3. ROUTE LEAKS ---
    crear_evento("TelekomMalaysia_Normal", "normal", 2015, 6, 11, 8, 11),
    crear_evento("TelekomMalaysia_Leak", "ataque", 2015, 6, 12, 8, 11),
    crear_evento("Google_MainOne_Normal", "normal", 2018, 11, 11, 20, 23),
    crear_evento("Google_MainOne_Leak", "ataque", 2018, 11, 12, 20, 23),
    crear_evento("Cloudflare_Normal", "normal", 2019, 6, 23, 10, 13),
    crear_evento("Cloudflare_Leak", "ataque", 2019, 6, 24, 10, 13),
    crear_evento("CenturyLink_Normal", "normal", 2020, 8, 29, 10, 13),
    crear_evento("CenturyLink_Leak", "ataque", 2020, 8, 30, 10, 13),
    crear_evento("Vodafone_India_Normal", "normal", 2021, 4, 15, 13, 16),
    crear_evento("Vodafone_India_Leak", "ataque", 2021, 4, 16, 13, 16),

    # --- 4. MASS OUTAGES ---
    crear_evento("Facebook_Normal", "normal", 2021, 10, 3, 15, 18),
    crear_evento("Facebook_Outage", "ataque", 2021, 10, 4, 15, 18),
    crear_evento("Rogers_Canada_Normal", "normal", 2022, 7, 7, 8, 11),
    crear_evento("Rogers_Canada_Outage", "ataque", 2022, 7, 8, 8, 11),
    crear_evento("KDDI_Japan_Normal", "normal", 2022, 6, 30, 16, 19),
    crear_evento("KDDI_Japan_Outage", "ataque", 2022, 7, 1, 16, 19),
    crear_evento("Spark_NZ_Normal", "normal", 2023, 9, 5, 14, 17),
    crear_evento("Spark_NZ_Outage", "ataque", 2023, 9, 6, 14, 17),
    crear_evento("Optus_Aus_Normal", "normal", 2023, 11, 7, 17, 20),
    crear_evento("Optus_Aus_Outage", "ataque", 2023, 11, 8, 17, 20)
])

# ==============================================================================
# DUAL PIPELINE EXECUTION
# ==============================================================================

# 1. RAW DATA DOWNLOAD
print("\n[1/3] Descargando datos crudos...")
utils.runJobs(dataset.getJobs(), folder + "collect_jobs")

# 2. TRANSFORMATION 1: Volumetric Pipeline
print("\n[2/3] Procesando familia de VOLUMEN (Features)...")
datTran_stats = DatasetTransformation(folder, "BML.transform", "Features")
datTran_stats.setParams({"global": {"Period": 5}})
utils.runJobs(datTran_stats.getJobs(), folder + "transform_jobs_stats")

# 3. TRANSFORMATION 2: Topological Pipeline
print("\n[3/3] Procesando familia de GRAFOS (GraphFeatures)...")
datTran_graphs = DatasetTransformation(folder, "BML.transform", "GraphFeatures")
datTran_graphs.setParams({"global": {"Period": 5}})
utils.runJobs(datTran_graphs.getJobs(), folder + "transform_jobs_graphs")

print("\nExtracción completada.")