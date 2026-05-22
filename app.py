import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# =========================================
# Page configuration
# =========================================
st.set_page_config(
    page_title="CAA Regression Calculator (RF)",
    layout="centered"
)

st.title("CAA Regression Prediction Calculator")

st.markdown(
    """
This web calculator predicts the probability of
**CAA regression** using a Random Forest model
and provides individualized SHAP interpretation.
"""
)

# =========================================
# Load RF model
# =========================================
@st.cache_resource
def load_model():
    return joblib.load("RF_model.pkl")

model = load_model()

# =========================================
# Sidebar input
# =========================================
st.sidebar.header("Patient Characteristics")

def user_input_features():

    # Age
    age = st.sidebar.number_input(
        "Age at diagnosis (months)",
        min_value=0.0,
        max_value=240.0,
        value=36.0
    )

    # Fever duration
    fever = st.sidebar.selectbox(
        "Fever duration >10 days",
        ["over 10 days", "within 10 days"]
    )

    over_10_days = 1 if fever == "over 10 days" else 0

    # IVIG response
    ivig = st.sidebar.selectbox(
        "IVIG treatment response",
        ["IVIG responsive", "IVIG resistant"]
    )

    ivig_resistance = 1 if ivig == "IVIG resistant" else 0

    # CAA classification
    caa = st.sidebar.selectbox(
        "CAA Classification",
        ["Small CAA", "Medium CAA", "Giant CAA"]
    )

    caa_map = {
        "Small CAA": 1,
        "Medium CAA": 2,
        "Giant CAA": 3
    }

    # WBC
    wbc = st.sidebar.number_input(
        "White blood cell count (×10⁹/L)",
        min_value=0.0,
        max_value=100.0,
        value=10.0
    )

    # Hb
    hb = st.sidebar.number_input(
        "Hemoglobin (g/L)",
        min_value=0.0,
        max_value=250.0,
        value=120.0
    )

    # PLT
    plt_count = st.sidebar.number_input(
        "Platelet count (×10⁹/L)",
        min_value=0.0,
        max_value=2000.0,
        value=350.0
    )

    # CRP
    crp = st.sidebar.number_input(
        "C-reactive protein (mg/L)",
        min_value=0.0,
        max_value=300.0,
        value=30.0
    )

    # PA
    pa = st.sidebar.number_input(
        "Prealbumin (mg/L)",
        min_value=0.0,
        max_value=500.0,
        value=130.0
    )

    # CST3
    cst3 = st.sidebar.number_input(
        "CST3 mRNA (2^⁻ΔΔCT)",
        min_value=0.0,
        max_value=20.0,
        value=1.2
    )

    # Create dataframe
    data = {
        "Age": age,
        "Fever_duration_over_10days": over_10_days,
        "IVIG_resistance": ivig_resistance,
        "Classification_of_CAA": caa_map[caa],
        "WBC": wbc,
        "Hb": hb,
        "PLT": plt_count,
        "CRP": crp,
        "PA": pa,
        "CST3mRNA": cst3
    }

    return pd.DataFrame(data, index=[0])

# =========================================
# User input
# =========================================
input_df = user_input_features()

# =========================================
# Prediction
# =========================================
pred_prob = model.predict_proba(input_df)[0, 1]
pred_class = model.predict(input_df)[0]

# =========================================
# Display prediction
# =========================================
st.subheader("Prediction Result")

st.metric(
    label="Predicted Probability of CAA Regression",
    value=f"{pred_prob * 100:.2f}%"
)

if pred_class == 1:
    st.success("Prediction: Regression")
else:
    st.error("Prediction: No Regression")

# =========================================
# SHAP explanation
# =========================================
st.subheader("Individualized SHAP Explanation")

# Create explainer
explainer = shap.Explainer(model)

# Calculate SHAP values
shap_values = explainer(input_df)

# =========================================
# Display-friendly feature names
# =========================================
FEATURE_NAME_MAP = {

    "Age": "Age at diagnosis (months)",

    "Fever_duration_over_10days":
        "Fever duration >10 days",

    "IVIG_resistance":
        "IVIG treatment resistance",

    "Classification_of_CAA":
        "CAA classification",

    "WBC":
        "White blood cell count (×10⁹/L)",

    "Hb":
        "Hemoglobin (g/L)",

    "PLT":
        "Platelet count (×10⁹/L)",

    "CRP":
        "C-reactive protein (mg/L)",

    "PA":
        "Prealbumin (mg/L)",

    "CST3mRNA":
        "CST3 mRNA (2^⁻ΔΔCT)"
}

# =========================================
# Rename dataframe columns
# =========================================
input_df_display = input_df.copy()

input_df_display.columns = [
    FEATURE_NAME_MAP[col]
    for col in input_df.columns
]

# =========================================
# Build new SHAP explanation
# =========================================
shap_value_single = shap.Explanation(

    values=shap_values.values[0, :, 1],

    base_values=shap_values.base_values[0, 1],

    data=input_df_display.iloc[0].values,

    feature_names=input_df_display.columns
)

# =========================================
# Waterfall plot
# =========================================
fig, ax = plt.subplots(figsize=(10, 6))

shap.plots.waterfall(
    shap_value_single,
    max_display=10,
    show=False
)

# Beautify labels
for label in ax.get_yticklabels():
    label.set_fontsize(10)

# Add probability text
ax.text(
    0.98,
    0.02,
    f"Predicted probability = {pred_prob * 100:.2f}%",
    transform=ax.transAxes,
    ha="right",
    fontsize=11,
    bbox=dict(
        facecolor="white",
        alpha=0.8
    )
)

plt.tight_layout()

# Show plot ONLY once
st.pyplot(fig)

# =========================================
# Finish
# =========================================
st.success("Prediction completed successfully.")





