# 🏥 AI-Powered Clinical Decision Support System for Gallstone Disease

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Live%20Demo-blue)]([LINK_TO_YOUR_HUGGINGFACE_SPACE_HERE](https://huggingface.co/spaces/Seafari/gallstone-risk-ai))
[![Paper](https://img.shields.io/badge/Paper-Procedia%20Computer%20Science-red)]()
[![Python](https://img.shields.io/badge/Python-3.9-yellow)]()

**Note**: This Project is the implemention of my accepted research paper on gallstone disease. 

**Official implementation of the research paper:** *"An Interpretable Machine Learning Framework for Gallstone Disease Prediction using XGBoost and Explainable AI"* 

**Status:** Accepted at ICMLDE 2025 | Published Link yet to be received*

---

## Project Overview
This repository hosts the **Interactive Clinical Decision Support System (CDSS)** developed to predict Gallstone Disease risk without expensive imaging. Unlike "black-box" AI models, this framework prioritizes **clinical interpretability** using SHAP (Shapley Additive exPlanations).

### Key Research Findings
* **Performance:** The XGBoost model achieved **85.9% Accuracy** and **0.86 AUC** on the test set, outperforming Random Forest and AdaBoost benchmarks.
* **Biomarkers:** Identified **C-Reactive Protein (CRP)** and **Vitamin D** as top predictors, validating known physiological links.
* **Novelty:** Discovered that **ECF/TBW Ratio** (Fluid Balance) is a critical novel predictor for gallstone formation.

---

## Features
* **Real-time Risk Stratification:** Instantly calculates disease probability based on patient vitals.
* **Local Interpretability (Force Plots):** Visualizes the "tug-of-war" between risk factors for individual patients.
* **Global Explainability:** Displays the overall feature importance ranking (CRP > Vitamin D > ECF/TBW).
* **Dockerized Deployment:** Fully containerized application running on Hugging Face Spaces.

---

## Tech Stack
* **Model:** XGBoost Classifier (Optimized via RandomizedSearchCV)
* **Explainability:** SHAP (TreeExplainer), Matplotlib
* **Web Framework:** Streamlit
* **Deployment:** Docker, Hugging Face Spaces

## Installation (Local)
To run this research demo on your local machine:

```bash
# 1. Clone the repository
git clone https://github.com/SEAFARI/gallstone-risk-ai.git

# 2. Navigate to directory
cd gallstone-risk-ai

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the App
streamlit run app.py
