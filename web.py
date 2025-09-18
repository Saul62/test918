import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import warnings
import sys

# Suppress noisy warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# NumPy compatibility for deprecated np.bool in some stacks
if not hasattr(np, 'bool'):
    np.bool = bool

# Page config
st.set_page_config(
    page_title="LASI Multimorbidity Prediction (Random Forest)",
    page_icon="🩺",
    layout="wide"
)

# Features used by the trained model (order matters)
FEATURES = [
    "self_rate_health", "BMI", "working_status",
    "adl", "urbanrural", "marriage", "age",
    "ph_activities", "pain", "household_income", "score"
]

# English labels and descriptions
FEATURE_LABELS = {
    "self_rate_health": "Self-rated Health",
    "BMI": "BMI Tier",
    "working_status": "Working",
    "adl": "ADL Limitation",
    "urbanrural": "Urban Residence",
    "marriage": "Married",
    "age": "Age",
    "ph_activities": "Physical Activity Level",
    "pain": "Pain",
    "household_income": "Household Income",
    "score": "Cognitive Score (normalized)",
}

FEATURE_DESC = {
    "self_rate_health": "Likert 1–5: Very poor/Poor/Fair/Good/Excellent (encoded 1–5)",
    "BMI": "0/1/2 map to Low/Medium/High",
    "working_status": "0/1 map to No/Yes",
    "adl": "0/1 map to No/Yes (any ADL limitation)",
    "urbanrural": "0/1 map to No/Yes (urban)",
    "marriage": "0/1 map to No/Yes (married)",
    "age": "Typical 50–116 in sample",
    "ph_activities": "0/1/2 map to Low/Medium/High",
    "pain": "0/1 map to No/Yes",
    "household_income": "0/1/2/3 map to Low/Lower-middle/Upper-middle/High",
    "score": "Model expects normalized [0,1]; the app converts raw score to normalized",
}

# Option sets and formatters
YES_NO_OPTIONS = [0, 1]
YES_NO_FMT = lambda x: "No" if x == 0 else "Yes"

LEVEL3_OPTIONS = [0, 1, 2]  # 0=Low, 1=Medium, 2=High
LEVEL3_FMT = lambda x: {0: "Low", 1: "Medium", 2: "High"}[x]

LIKERT5_OPTIONS = [1, 2, 3, 4, 5]
LIKERT5_FMT = lambda x: {1: "Very poor", 2: "Poor", 3: "Fair", 4: "Good", 5: "Excellent"}[x]

INCOME_OPTIONS = [0, 1, 2, 3]
INCOME_FMT = lambda x: {0: "Low", 1: "Lower-middle", 2: "Upper-middle", 3: "High"}[x]


# Load model with numpy._core fallback compatibility for some environments
@st.cache_resource
def load_model():
    model_path = 'lasi_result.pkl'
    try:
        return joblib.load(model_path)
    except ModuleNotFoundError as e:
        if 'numpy._core' in str(e):
            import numpy as _np
            sys.modules['numpy._core'] = _np.core
            sys.modules['numpy._core._multiarray_umath'] = _np.core._multiarray_umath
            sys.modules['numpy._core.multiarray'] = _np.core.multiarray
            sys.modules['numpy._core.umath'] = _np.core.umath
            return joblib.load(model_path)
        raise


def main():
    st.sidebar.title("LASI Multimorbidity Prediction (Random Forest)")
    st.sidebar.markdown(
        "- Predicts presence of multimorbidity (0=No, 1=Yes) using 11 features.\n"
        "- Binary classification model (Random Forest).\n"
        "- Binary fields are shown as Yes/No; 3-level fields as Low/Medium/High.\n"
        "- Self-rated health 1–5 shown as Very poor → Excellent.\n"
        "- Cognitive score (score) is entered as a raw value and normalized internally."
    )

    with st.sidebar.expander("Features & Notes"):
        for k in FEATURES:
            st.markdown(f"- {FEATURE_LABELS.get(k,k)}: {FEATURE_DESC.get(k,'')}")

    # Fixed raw score range for normalization (hidden from UI)
    SCORE_RAW_MIN = 0
    SCORE_RAW_MAX = 39

    # Load model
    try:
        model = load_model()
        st.sidebar.success("Model loaded successfully")
    except Exception as e:
        st.sidebar.error(f"Failed to load model: {e}")
        return

    st.title("LASI Multimorbidity Risk Prediction")
    st.markdown("Enter the inputs below and click Predict.")

    col1, col2, col3 = st.columns(3)

    with col1:
        self_rate_health = st.selectbox(
            FEATURE_LABELS['self_rate_health'], LIKERT5_OPTIONS, format_func=LIKERT5_FMT
        )
        BMI = st.selectbox(
            FEATURE_LABELS['BMI'], LEVEL3_OPTIONS, format_func=LEVEL3_FMT
        )
        working_status = st.selectbox(
            FEATURE_LABELS['working_status'], YES_NO_OPTIONS, format_func=YES_NO_FMT
        )
        adl = st.selectbox(
            FEATURE_LABELS['adl'], YES_NO_OPTIONS, format_func=YES_NO_FMT
        )

    with col2:
        urbanrural = st.selectbox(
            FEATURE_LABELS['urbanrural'], YES_NO_OPTIONS, format_func=YES_NO_FMT
        )
        marriage = st.selectbox(
            FEATURE_LABELS['marriage'], YES_NO_OPTIONS, format_func=YES_NO_FMT
        )
        age = st.number_input(
            FEATURE_LABELS['age'], min_value=50, max_value=116, value=65, step=1
        )
        ph_activities = st.selectbox(
            FEATURE_LABELS['ph_activities'], LEVEL3_OPTIONS, format_func=LEVEL3_FMT
        )

    with col3:
        pain = st.selectbox(
            FEATURE_LABELS['pain'], YES_NO_OPTIONS, format_func=YES_NO_FMT
        )
        household_income = st.selectbox(
            FEATURE_LABELS['household_income'], INCOME_OPTIONS, format_func=INCOME_FMT
        )
        score_raw = st.slider(
            "Cognitive score (raw)",
            min_value=int(SCORE_RAW_MIN),
            max_value=int(SCORE_RAW_MAX),
            value=int((SCORE_RAW_MIN + SCORE_RAW_MAX) // 2),
            step=1,
        )
        # Normalize to [0,1]
        score_norm = (score_raw - SCORE_RAW_MIN) / (SCORE_RAW_MAX - SCORE_RAW_MIN)

    if st.button("Predict"):
        # Assemble input in the exact training order
        row = [
            self_rate_health, BMI, working_status,
            adl, urbanrural, marriage, age,
            ph_activities, pain, household_income, score_norm
        ]
        input_df = pd.DataFrame([row], columns=FEATURES)

        try:
            proba = model.predict_proba(input_df)[0]
            pred = int(model.predict(input_df)[0])
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

        # Minimal text only; no tables or extra charts
        st.subheader("Prediction Result")
        st.markdown(f"Predicted: {'Yes' if pred==1 else 'No'} (multimorbidity).  Probabilities – No: {proba[0]:.4f}, Yes: {proba[1]:.4f}")

        # SHAP explainability
        st.write("---")
        st.subheader("Explainability (SHAP)")
        try:
            explainer = shap.TreeExplainer(model)
            sv = explainer.shap_values(input_df)

            # Handle different SHAP return formats
            if isinstance(sv, list):
                shap_value = np.array(sv[1][0])  # class 1 contribution
                expected_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
            elif isinstance(sv, np.ndarray) and sv.ndim == 2:
                shap_value = sv[0]
                expected_value = explainer.expected_value
            elif isinstance(sv, np.ndarray) and sv.ndim == 3:
                shap_value = sv[0, :, 1]
                expected_value = explainer.expected_value[1]
            else:
                raise RuntimeError("Unrecognized SHAP output format")

            # Waterfall plot
            try:
                fig = plt.figure(figsize=(10, 7))
                shap.waterfall_plot(
                    shap.Explanation(
                        values=shap_value,
                        base_values=expected_value,
                        data=input_df.iloc[0].values,
                        feature_names=[FEATURE_LABELS.get(f, f) for f in FEATURES]
                    ),
                    max_display=11,
                    show=False
                )
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.error(f"Waterfall plot failed: {e}")

            # Force plot
            try:
                force_plot = shap.force_plot(
                    expected_value,
                    shap_value,
                    input_df.iloc[0],
                    feature_names=[FEATURE_LABELS.get(f, f) for f in FEATURES],
                    matplotlib=True,
                    show=False,
                    figsize=(20, 3)
                )
                st.pyplot(force_plot)
            except Exception as e:
                st.error(f"Force plot failed: {e}")
        except Exception as e:
            st.warning(f"Could not generate SHAP explanation: {e}")

        # No other tables/plots required


if __name__ == "__main__":
    main()
