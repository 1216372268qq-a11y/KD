import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from io import BytesIO

# ==================================================
# Page configuration
# ==================================================
st.set_page_config(
    page_title="CAA Regression Prediction Calculator",
    layout="centered"
)

st.title("CAA Regression Prediction Calculator")
st.markdown(
    """
    This clinical decision-support tool estimates the probability of  
    **coronary artery aneurysm (CAA) regression** and provides an  
    individualized, transparent SHAP-based explanation.
    """
)

# ==================================================
# Upload model & background data (PRIVATE)
# ==================================================
st.subheader("Upload Required Files")

model_file = st.file_uploader(
    "Upload trained XGBoost model (.pkl)",
    type=["pkl"]
)

data_file = st.file_uploader(
    "Upload background dataset (.csv, used only for SHAP reference)",
    type=["csv"]
)

if model_file is None or data_file is None:
    st.info("Please upload both the model file and the background dataset to proceed.")
    st.stop()

# Load model
model = joblib.load(BytesIO(model_file.read()))

# Load background data
X_background = pd.read_csv(BytesIO(data_file.read()))
X_background = X_background.drop(columns=["CAA_regression"])

# ==================================================
# Sidebar – Patient input
# ==================================================
st.sidebar.header("Patient Characteristics")

def user_input_features():
    age = st.sidebar.number_input(
        "Age at diagnosis (months)", 1, 240, 36
    )

    bmi = st.sidebar.number_input(
        "Body mass index (kg/m²)", 5.0, 40.0, 16.0
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
        "Platelet count (×10¹²/L)", 1, 2000, 350
    )

    esr = st.sidebar.number_input(
        "Erythrocyte sedimentation rate (mm/h)", 1, 200, 50
    )

    pa = st.sidebar.number_input(
        "Prealbumin (mg/L)", 0, 500, 130
    )

    cst3 = st.sidebar.number_input(
        "CST3 mRNA (2⁻ΔΔCT)", 0.0, 10.0, 1.2
    )

    return pd.DataFrame([{
        "Age": age,
        "BMI": bmi,
        "IVIG_resistance": ivig_resistance,
        "Classification_of_CAA": caa_class,
        "PLT": plt_count,
        "ESR": esr,
        "PA": pa,
        "CST3mRNA": cst3
    }])

input_df = user_input_features()

# Encoding (consistent with training)
caa_map = {"Small CAA": 1, "Medium CAA": 2, "Giant CAA": 3}
input_df["Classification_of_CAA"] = input_df["Classification_of_CAA"].map(caa_map)

# ==================================================
# Prediction
# ==================================================
pred_prob = model.predict_proba(input_df)[0, 1]
pred_class = model.predict(input_df)[0]

st.subheader("Prediction Result")
st.metric(
    "Predicted probability of CAA regression",
    f"{pred_prob * 100:.2f}%"
)

st.write(
    "**Model output:**",
    "Regression" if pred_class == 1 else "No regression"
)

# ==================================================
# SHAP – Waterfall plot (CLEAN & BEAUTIFUL)
# ==================================================
st.subheader("Individualized Model Explanation (SHAP Waterfall Plot)")

explainer = shap.TreeExplainer(model, X_background)
shap_value = explainer(input_df)

# Display-friendly feature names
FEATURE_NAME_MAP = {
    "Age": "Age at diagnosis (months)",
    "BMI": "Body mass index (kg/m²)",
    "IVIG_resistance": "IVIG resistance",
    "Classification_of_CAA": "CAA classification",
    "PLT": "Platelet count (×10¹²/L)",
    "ESR": "Erythrocyte sedimentation rate (mm/h)",
    "PA": "Prealbumin (mg/L)",
    "CST3mRNA": "CST3 mRNA (2⁻ΔΔCT)"
}

shap_display = shap.Explanation(
    values=shap_value.values,
    base_values=shap_value.base_values,
    data=input_df.values,
    feature_names=[FEATURE_NAME_MAP[c] for c in input_df.columns]
)

# ---- Let SHAP create figure ----
shap.plots.waterfall(shap_display[0], show=False)

fig = plt.gcf()
fig.set_size_inches(10, 7)

ax = plt.gca()
for label in ax.get_yticklabels():
    label.set_fontsize(10)

ax.text(
    0.99, 0.02,
    f"Predicted probability = {pred_prob * 100:.2f}%",
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=11,
    bbox=dict(facecolor="white", alpha=0.85, edgecolor="gray")
)

plt.tight_layout()
st.pyplot(fig)
plt.close()

st.success("Prediction and explanation completed successfully.")
