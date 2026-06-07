# BGP Anomaly Detection: Topological vs. Volumetric Paradigms

This repository contains the code, data extraction pipelines, and Machine Learning models developed for my Bachelor's Thesis on Border Gateway Protocol (BGP) anomaly detection. 

The project empirically evaluates and demonstrates a paradigm shift in network security: proving that extracting structural/topological features (using mathematical graphs) from BGP traffic yields a fundamentally superior detection space compared to traditional volumetric message counting, effectively detecting silent Outages and subtle Hijacks that standard models miss.

## Key Highlights
* **Dual Pipeline Comparison:** Direct empirical comparison between Volumetric features (message counts) and Topological features (PageRank, assortativity, centrality).
* **Robust ML Evaluation:** Utilizes XGBoost, Random Forest, MLP, and Logistic Regression with Stratified K-Fold cross-validation to prevent data leakage.
* **Event-Isolated Z-Score:** Implements a strict normalization methodology to isolate anomalous traffic from background noise and mitigate the 'Year Effect'.
* **Explainable AI (XAI):** Uses SHAP (SHapley Additive exPlanations) to mathematically deconstruct the model's decision logic, proving that graph deformation (specifically PageRank) is the true signature of BGP anomalies.

## Repository Structure

The workflow is divided into 5 main scripts that must be executed sequentially:

1. 1_data_extraction.py: Utilizes the BML framework and RIPE RIS collectors (rrc04, rrc06) to download raw BGP data spanning 15 years and extracts both Volumetric and Graph features for 40 specific incidents (Normal vs. Attack windows).
2. 2_data_consolidation.py: A custom parser that navigates the BML output directories, corrects transposition artifacts, injects semantic metadata, and consolidates the JSONs into 4 core .csv datasets.
3. 3_model_evaluation.py: The core ML training script. Applies event-isolated Z-Score normalization, performs 5-Fold cross-validation, and extracts global and category-specific performance metrics.
4. 4_shap_explainability.py: Trains the final XGBoost topological model and applies SHAP TreeExplainer to generate global and local feature importance plots, providing full transparency into the algorithm's decisions.
5. 5_confusion_matrix.py: Dedicated visualization script that generates high-resolution, publication-ready Confusion Matrix heatmaps (PDF) for both paradigms, ensuring the exact same data partition used in the SHAP analysis.

## Requirements & Installation

The code is written in Python 3.x. To install the required dependencies, run the following command in your terminal:

    pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn

*(Note: The BML framework for data extraction requires specific system dependencies. Refer to the official BML documentation if you intend to re-run the 1_data_extraction.py script).*

## How to Run

1. Ensure the generated datasets (volumen_rrc04.csv, grafos_rrc04.csv, etc.) are placed inside a folder named dataset_csvs/ in the root directory.

2. Run the evaluation pipeline to extract the cross-validation metrics:
    python 3_model_evaluation.py

3. Run the Explainable AI analysis to extract the SHAP plots:
    python 4_shap_explainability.py

4. Generate the comparative visual matrices:
    python 5_confusion_matrix.py

The scripts will automatically output high-resolution .pdf and .png figures directly into the root folder.

## License
This project was developed as a Bachelor's Thesis. Feel free to explore, fork, and use the code for academic or research purposes.