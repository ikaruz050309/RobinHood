import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from data_cleaning import dataset_cleaning, validate_dataset
from model import model_training
from explainability import captum_explainability
from llm_explanation import generate_llm_explication, DEFAULT_MODEL_NAME

st.set_page_config(page_title="Time Series Forecasting + XAI", page_icon="📈", layout="wide")

st.title("📈 Time Series Forecasting with Explainable AI")
st.caption(
    "Pipeline: data cleaning → TCN + LSTM + Attention model → "
    "explainability (Captum) → natural-language explanation (Groq LLM)"
)

st.markdown(
    "This tool was built to help you **understand *why* an AI model makes a given "
    "prediction** on financial time series, not just to output a number. By pairing "
    "the forecast with Captum feature attributions and a plain-language explanation, "
    "the goal is to make the model's reasoning inspectable rather than a black box. "
    "In the near future, the ambition is to grow this into a genuine **AI financial "
    "advisor** — today it is an educational/experimental prototype, not that yet."
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
for key in [
    "raw_data", "clean_df", "model", "predictions",
    "X_test_t", "dataframes", "importance_map", "explanation",
]:
    if key not in st.session_state:
        st.session_state[key] = None

# ---------------------------------------------------------------------------
# 1. Data upload
# ---------------------------------------------------------------------------
st.header("1. Upload your data")

st.warning(
    "**Before you upload — please read:**\n"
    "- 📊 **The richer the dataset, the better the prediction.** More numeric "
    "columns (price, volume, indicators, etc.) and more history generally give "
    "the model more signal to learn from.\n"
    "- 📅 **Adding a date column as the first column is recommended.** It lets "
    "the pipeline sort the data chronologically and encode seasonality.\n"
    "- 🚫 **A `ticker` column will be rejected.** It usually means multiple "
    "companies/assets are mixed in the same file.\n"
    "- 🏢 **The dataset must contain data for ONE company only.** This pipeline "
    "does not support multi-company/multi-asset files."
)

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        candidate_data = pd.read_csv(uploaded_file)
        validate_dataset(candidate_data)
        st.session_state.raw_data = candidate_data
        st.success(f"File loaded: {candidate_data.shape[0]} rows, {candidate_data.shape[1]} columns")
        st.dataframe(candidate_data.head(), use_container_width=True)
    except ValueError as ve:
        st.session_state.raw_data = None
        st.error(f"Dataset rejected: {ve}")
    except Exception as e:
        st.session_state.raw_data = None
        st.error(f"Error while reading the CSV: {e}")

# ---------------------------------------------------------------------------
# 2. Data cleaning
# ---------------------------------------------------------------------------
if st.session_state.raw_data is not None:
    st.header("2. Clean the data")
    if st.button("🧹 Clean the data", type="primary"):
        with st.spinner("Cleaning in progress..."):
            try:
                clean_df, original = dataset_cleaning(st.session_state.raw_data)
                st.session_state.clean_df = clean_df
                st.success(f"Data cleaned: {clean_df.shape[0]} rows, {clean_df.shape[1]} columns")
            except Exception as e:
                st.error(f"Error while cleaning: {e}")

    if st.session_state.clean_df is not None:
        st.dataframe(st.session_state.clean_df.head(10), use_container_width=True)

# ---------------------------------------------------------------------------
# 3. Model training
# ---------------------------------------------------------------------------
if st.session_state.clean_df is not None:
    st.header("3. Train the model")

    n_rows = len(st.session_state.clean_df)
    col1, col2 = st.columns(2)
    with col1:
        N = st.number_input(
            "History window size (N)", min_value=1,
            value=min(10, max(1, n_rows // 4)), step=1,
        )
    with col2:
        K = st.number_input("Prediction horizon (K)", min_value=1, value=3, step=1)

    st.caption(
        "ℹ️ If N + K exceeds the number of available rows, synthetic rows will be "
        "generated using Gaussian noise (AWGN) to fill the dataset."
    )

    if st.button("🚀 Train the model", type="primary"):
        with st.spinner("Training in progress (50 epochs)... this may take a few minutes."):
            try:
                model, predictions, X_test_t, dataframes = model_training(
                    st.session_state.clean_df, N=int(N), K=int(K)
                )
                st.session_state.model = model
                st.session_state.predictions = predictions
                st.session_state.X_test_t = X_test_t
                st.session_state.dataframes = dataframes
                # reset downstream steps
                st.session_state.importance_map = None
                st.session_state.explanation = None
                st.success("Model trained successfully ✅")
            except Exception as e:
                st.error(f"Error during training: {e}")

# ---------------------------------------------------------------------------
# 4. Result: chart + explainability + AI explanation, all in one place
# ---------------------------------------------------------------------------
if st.session_state.predictions is not None:
    st.header("4. Result")

    preds = st.session_state.predictions
    feature_names = st.session_state.dataframes.select_dtypes(include=[np.number]).columns.tolist()
    n_features = preds.shape[-1]

    st.subheader("Prediction chart")
    feature_idx = st.selectbox(
        "Variable to display",
        options=list(range(n_features)),
        format_func=lambda i: feature_names[i] if i < len(feature_names) else f"variable_{i}",
    )

    fig, ax = plt.subplots()
    ax.plot(preds[:, 0, feature_idx], marker="o", markersize=3, label="Prediction (1st step of the horizon)")
    ax.set_xlabel("Test sample")
    ax.set_ylabel("Normalized value")
    ax.legend()
    st.pyplot(fig)

    st.caption(
        "⚠️ Values are normalized (MinMaxScaler 0-1): the inverse scaler is not "
        "returned by the original pipeline, so the displayed values are not the "
        "real-world price/units."
    )

    st.divider()
    st.subheader("Generate the full result (explainability + AI summary)")
    st.caption(
        "This computes Captum feature attributions and calls the Groq API to "
        "produce a plain-language explanation, displayed directly below."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        model_name = st.text_input("Groq model", value=DEFAULT_MODEL_NAME)
    with col2:
        groq_key_input = st.text_input(
            "Groq API key (optional)", type="password",
            help="Only needed if GROQ_API_KEY is not already set via environment "
                 "variable or Streamlit secrets. Get a free key at "
                 "https://console.groq.com/keys",
        )

    if st.button("🧠 Generate result", type="primary"):
        with st.spinner("Computing feature importance..."):
            try:
                importance_map = captum_explainability(
                    st.session_state.model, st.session_state.X_test_t, st.session_state.dataframes
                )
                st.session_state.importance_map = importance_map
            except Exception as e:
                st.error(f"Error during Captum computation: {e}")
                importance_map = None

        if importance_map is not None:
            api_key = groq_key_input or os.environ.get("GROQ_API_KEY")
            if not api_key:
                try:
                    api_key = st.secrets.get("GROQ_API_KEY")
                except Exception:
                    api_key = None

            with st.spinner("Generating the AI explanation via Groq..."):
                try:
                    explanation = generate_llm_explication(
                        importance_map,
                        st.session_state.predictions,
                        st.session_state.dataframes,
                        model_name=model_name,
                        api_key=api_key,
                    )
                    st.session_state.explanation = explanation
                    st.success("Result generated ✅")
                except Exception as e:
                    st.error(f"Error during Groq generation: {e}")

    # Below-the-graph result: text explanation + supporting importance chart
    if st.session_state.explanation is not None:
        st.markdown("### 🧠 AI-generated analysis")
        st.write(st.session_state.explanation)

    if st.session_state.importance_map is not None:
        st.markdown("### 🔍 Feature importance (Captum — Integrated Gradients)")
        imp_df = pd.DataFrame(
            list(st.session_state.importance_map.items()),
            columns=["Variable", "Importance (%)"]
        ).sort_values("Importance (%)", ascending=False)
        st.dataframe(imp_df, use_container_width=True)
        st.bar_chart(imp_df.set_index("Variable"))

# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------
st.divider()
if st.button("🔄 Reset the whole pipeline"):
    for key in [
        "raw_data", "clean_df", "model", "predictions",
        "X_test_t", "dataframes", "importance_map", "explanation",
    ]:
        st.session_state[key] = None
    st.rerun()
