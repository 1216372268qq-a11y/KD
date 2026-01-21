import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt

# ===============================
# Page configuration
# ===============================
st.set_page_config(
    page_title="CAA Regression Prediction Calculator",
    layout="centered"
)

st.title("CAA Regression Prediction Calculator")
st.markdown(
    "This clinical decision-support tool predicts the probability of "
    "**coronary artery aneurysm (CAA) regression** and provides an "
    "individualized SHAP-based explanation."
)

# ===============================
# Load trained model
# ===============================
@st.cache_resource
def load_model():
    return joblib.load("xgb_caa_regression.pkl")

model = load_model()

# ===============================
# Sidebar – Patient input
# ===============================
st.sidebar.header("Patient Characteristics")

def user_input_features():
    age = st.sidebar.number_input(
        "Age at diagnosis (months)",
        min_value=1,
        max_value=240,
        value=36
    )

    bmi = st.sidebar.number_input(
        "Body mass index (kg/m²)",
        min_value=5.0,
        max_value=40.0,
        value=16.0
    )

    ivig_status = st.sidebar.selectbox(
        "IVIG treatment response",
        ["IVIG responsive", "IVIG resistant"]
    )
    ivig_resistance = 1 if ivig_status == "IVIG resistant" else 0

    caa_class = st.sidebar.selectbox(
        "CAA classification",
        ["Small CAA", "Medium CAA", "Giant CAA"]
    )

    plt_count = st.sidebar.number_input(
        "Platelet count (×10⁹/L)",
        min_value=1,
        max_value=2000,
        value=350
    )

    esr = st.sidebar.number_input(
        "Erythrocyte sedimentation rate (mm/h)",
        min_value=1,
        max_value=200,
        value=50
    )

    pa = st.sidebar.number_input(
        "Prealbumin (mg/L)",
        min_value=0,
        max_value=500,
        value=130
    )

    cst3 = st.sidebar.number_input(
        "CST3 mRNA (2^⁻ΔΔCT)",
        min_value=0.0,
        max_value=10.0,
        value=1.2
    )

    data = {
        "Age": age,
        "BMI": bmi,
        "IVIG_resistance": ivig_resistance,
        "Classification_of_CAA": caa_class,
        "PLT": plt_count,
        "ESR": esr,
        "PA": pa,
        "CST3mRNA": cst3
    }

    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# ===============================
# Encoding (consistent with training)
# ===============================
caa_map = {
    "Small CAA": 1,
    "Medium CAA": 2,
    "Giant CAA": 3
}
input_df["Classification_of_CAA"] = input_df["Classification_of_CAA"].map(caa_map)

# ===============================
# Prediction
# ===============================
pred_prob = model.predict_proba(input_df)[0, 1]
pred_class = model.predict(input_df)[0]

st.subheader("Prediction Result")
st.metric(
    label="Predicted probability of CAA regression",
    value=f"{pred_prob * 100:.2f}%"
)

st.write(
    "**Model output:**",
    "Regression" if pred_class == 1 else "No regression"
)

# ===============================
# SHAP – Waterfall plot (beautified)
# ===============================
st.subheader("Individualized Model Explanation (SHAP Waterfall Plot)")

# Load background data
explainer = shap.TreeExplainer(model)
shap_value = explainer(input_df)

FEATURE_NAME_MAP = {
    "Age": "Age at diagnosis (months)",
    "BMI": "Body mass index (kg/m²)",
    "IVIG_resistance": "IVIG resistance status",
    "Classification_of_CAA": "CAA classification",
    "PLT": "Platelet count (×10⁹/L)",
    "ESR": "Erythrocyte sedimentation rate (mm/h)",
    "PA": "Prealbumin (mg/L)",
    "CST3mRNA": "CST3 mRNA (2⁻ΔΔCT)"
}

input_df_display = input_df.copy()
input_df_display.columns = [FEATURE_NAME_MAP[col] for col in input_df.columns]

shap_value_display = shap.Explanation(
    values=shap_value.values,
    base_values=shap_value.base_values,
    data=input_df_display.values,
    feature_names=input_df_display.columns
)


explainer = shap.TreeExplainer(model)
shap_value = explainer(input_df)

# Display-friendly feature names
FEATURE_NAME_MAP = {
    "Age": "Age at diagnosis (months)",
    "BMI": "Body mass index (kg/m²)",
    "IVIG_resistance": "IVIG resistance status",
    "Classification_of_CAA": "CAA classification",
    "PLT": "Platelet count (×10⁹/L)",
    "ESR": "Erythrocyte sedimentation rate (mm/h)",
    "PA": "Prealbumin (mg/L)",
    "CST3mRNA": "CST3 mRNA (2^⁻ΔΔCT)"
}

input_df_display = input_df.copy()
input_df_display.columns = [
    FEATURE_NAME_MAP[col] for col in input_df.columns
]

shap_value_display = shap.Explanation(
    values=shap_value.values,
    base_values=shap_value.base_values,
    data=input_df_display.values,
    feature_names=input_df_display.columns
)

# -------- Plot with layout optimization --------
fig, ax = plt.subplots(figsize=(9, 6.5))

shap.plots.waterfall(
    shap_value_display[0],
    show=False
)

# Reduce y-axis label font size
for label in ax.get_yticklabels():
    label.set_fontsize(10)

# Add prediction text
ax.text(
    0.99, 0.02,
    f"Predicted probability = {pred_prob * 100:.2f}%",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=11,
    bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray")
)

plt.tight_layout(rect=[0, 0.02, 1, 0.98])
st.pyplot(fig)

st.success("Prediction completed successfully.")





