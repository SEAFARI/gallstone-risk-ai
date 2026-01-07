import streamlit as st
import numpy as np
import pandas as pd
import xgboost as xgb
import shap
from streamlit_shap import st_shap
import matplotlib.pyplot as plt


# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Gallstone Risk AI", 
    page_icon="🏥",
    layout="wide"
)

# --- 2. SETUP FEATURE NAMES & AVERAGES ---
# Based on your provided "Index Mapping"
FEATURE_NAMES = [
    "Age", "Gender", "Comorbidity", "CAD", "Hypothyroidism", "Hyperlipidemia", 
    "Diabetes Mellitus", "Height", "Weight", "BMI", "Total Body Water", 
    "Extracellular Water", "Intracellular Water", "ECF/TBW Ratio (%)", "Total Body Fat Ratio", 
    "Lean Mass (%)", "Body Protein (%)", "Visceral Fat Rating", "Bone Mass", 
    "Muscle Mass", "Obesity (%)", "Total Fat Content", "Visceral Fat Area", 
    "Visceral Muscle Area", "Hepatic Fat Accumulation", "Glucose", "Total Cholesterol", 
    "LDL", "HDL", "Triglyceride", "AST", "ALT", "ALP", "Creatinine", 
    "GFR", "C-Reactive Protein (CRP)", "Hemoglobin", "Vitamin D"
]


AVERAGE_VALUES = [
    48.06, 0.49, 0.33, 0.037, 0.028, 0.025, 0.13, 167.15, 80.56, 28.87, 
    40.58, 17.07, 23.63, 42.21, 28.27, 71.63, 15.93, 9.07, 2.80, 54.27, 
    35.85, 23.48, 12.17, 30.40, 1.15, 108.68, 203.49, 126.65, 49.47, 
    144.50, 21.68, 26.85, 73.11, 0.80, 100.81, 1.85, 14.41, 21.40
]

# --- 3. LOAD MODEL ---
@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model("model.json")
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"[ERROR] Error loading 'model.json'. Make sure it is in the same folder.\nDetails: {e}")
    st.stop()

# --- 4. HEADER ---
st.title("AI-Powered Gallstone Disease Risk Prediction")
st.markdown("""
    **Research Demo:** Interpretable Machine Learning Framework for Gallstone Disease Prediction.
    *Adjust the clinical parameters in the sidebar to analyze risk factors.*
""")
st.divider()

# --- 5. SIDEBAR & INPUT LOGIC ---
st.sidebar.header("Patient Vitals")

def get_user_input():
    # Start with a copy of the Average Patient (so hidden features aren't 0)
    input_data = list(AVERAGE_VALUES)
    
    # --- KEY RISK FACTORS (From your Paper) ---
    st.sidebar.subheader("Key Inflammatory & Metabolic Markers")
    
    # Index 35: CRP (Default: 1.85)
    crp = st.sidebar.slider("C-Reactive Protein (CRP)", 0.0, 50.0, 1.85, help="Top predictor in study.")
    input_data[35] = crp
    
    # Index 37: Vitamin D (Default: 21.40)
    vit_d = st.sidebar.slider("Vitamin D", 0.0, 100.0, 21.40)
    input_data[37] = vit_d

    # Index 13: ECF/TBW Ratio (Default: 42.21%)
    # Note: Your average is 42.21, so this is likely a percentage (0-100), not a decimal (0.42).
    ecf_tbw = st.sidebar.slider("ECF/TBW Ratio (%)", 30.0, 60.0, 42.21)
    input_data[13] = ecf_tbw

    # --- DEMOGRAPHICS ---
    st.sidebar.subheader("Demographics & Body Comp")
    
    # Index 0: Age (Default: 48)
    age = st.sidebar.slider("Age", 18, 90, 48)
    input_data[0] = age
    
    # Index 20: Obesity % (Default: 35.85)
    obesity = st.sidebar.slider("Obesity (%)", 10.0, 70.0, 35.85)
    input_data[20] = obesity

    # Index 1: Gender (Default 0.49 -> Round to 0 or 1)
    gender_label = st.sidebar.selectbox("Gender", ["Female", "Male"])
    input_data[1] = 1.0 if gender_label == "Male" else 0.0

    # Convert to 2D numpy array for XGBoost: (1, 38)
    features = np.array([input_data])
    return features, crp, vit_d, ecf_tbw

# Get input from user
input_features, val_crp, val_vitd, val_ecf = get_user_input()

# --- 6. PREDICTION ENGINE & ANALYSIS ---
if st.button("Run Clinical Analysis", type="primary"):
    
    # 1. PREDICT
    prediction_class = model.predict(input_features)[0]
    prediction_probs = model.predict_proba(input_features)
    prob_disease = prediction_probs[0][1] * 100

    # 2. CALCULATE SHAP (Do this once)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_features)
    
    # --- SECTION 1: DIAGNOSTIC RESULTS ---
    st.divider()
    col_res, col_gauge = st.columns([2, 1])
    
    with col_res:
        st.subheader("Diagnostic Result")
        if prediction_class == 1:
            st.error("🔴 **POSITIVE** Prediction")
            st.metric("Disease Probability", f"{prob_disease:.1f}%", delta="High Risk")
            st.warning(f"Patient flagged for Gallstone Disease.")
        else:
            st.success("🟢 **NEGATIVE** Prediction")
            st.metric("Disease Probability", f"{prob_disease:.1f}%", delta="-Low Risk")
            st.info("Patient is likely healthy.")

    # --- SECTION 2: RESEARCH PLOTS (The "Three Plots") ---
    st.divider()
    st.subheader("🔍 Explainable AI (XAI) Analysis")
    st.info("Visualizing decision logic based on Kothari et al. (2025) framework.")

    tab1, tab2, tab3 = st.tabs(["1. Force Plot (Local)", "2. Decision Factors (LIME-Analysis)", "3. Global Importance"])

    # PLOT 1: SHAP FORCE PLOT (Visual "Tug-of-War")
    with tab1:
        st.write("**What this shows:** How each feature pushes the risk score higher (Red) or lower (Blue) from the baseline.")
        st_shap(shap.force_plot(
            explainer.expected_value, 
            shap_values[0,:], 
            input_features[0,:], 
            feature_names=FEATURE_NAMES
        ))
        
    

    # PLOT 2: CUSTOM BAR CHART (The LIME Replacement)
    with tab2:
        st.write("**What this shows:** The exact numerical contribution of the top features to this specific prediction.")
        
        # Create a DataFrame for the bar chart
        df_shap = pd.DataFrame({
            "Feature": FEATURE_NAMES,
            "Contribution": shap_values[0]
        })
        # Sort by absolute contribution (Impact)
        df_shap["Abs_Contribution"] = df_shap["Contribution"].abs()
        df_shap = df_shap.sort_values("Abs_Contribution", ascending=True).head(10) # Top 10 only
        
        # Color coding: Red for Risk, Blue for Protective
        colors = ["#FF4B4B" if x > 0 else "#1E90FF" for x in df_shap["Contribution"]]
        
        # Plot using Matplotlib (cleaner control than st.bar_chart)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(df_shap["Feature"], df_shap["Contribution"], color=colors)
        ax.set_xlabel("Impact on Risk Score")
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        st.pyplot(fig)
        
        

    # PLOT 3: GLOBAL IMPORTANCE (General Knowledge)
    with tab3:
        st.write("**What this shows:** Which features are most important *overall* for this model (not just this patient).")
        
        # Get feature importance from XGBoost model
        importance = model.feature_importances_
        indices = np.argsort(importance)[-10:] # Top 10
        
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.barh(range(10), importance[indices], color="#555555")
        ax2.set_yticks(range(10))
        ax2.set_yticklabels(np.array(FEATURE_NAMES)[indices])
        ax2.set_xlabel("Relative Importance (Gain)")
        st.pyplot(fig2)
        
        
        
    st.write(f"**CRP Level:** {val_crp} (Study found high CRP > 1.64 to be a primary driver)")
    st.write(f"**Vitamin D:** {val_vitd} (Study correlates low Vitamin D with higher risk)")
    st.write(f"**ECF/TBW:** {val_ecf}% (Fluid regulation impacts risk)")

else:
    st.info("Adjust patient vitals in the sidebar and click 'Run Clinical Analysis'.")