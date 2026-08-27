import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from captum.attr import IntegratedGradients
from groq import Groq

DEFAULT_GROQ_MODEL = "deepseek-r1-distill-llama-70b"

st.set_page_config(page_title="Time Series Forecasting + XAI", page_icon="📈", layout="wide")

st.title("📈 Time Series Forecasting with Explainable AI")
st.caption(
    "Pipeline: data cleaning → TCN + LSTM + Attention model → "
    "explainability (Captum) → natural-language explanation (Groq LLM)"
)
st.markdown(
    "This tool was built to help you **understand *why* an AI model makes a given "
    "prediction** on financial time series, not just to output a number. In the near "
    "future, the ambition is to grow this into a genuine **AI financial advisor** — "
    "today it is an educational/experimental prototype, not that yet."
)


# ---------------------------------------------------------------------------
# Pipeline functions
# ---------------------------------------------------------------------------

def validate_dataset(df):
    """Rejects datasets that mix several companies (a 'ticker' column)."""
    lower_cols = [str(c).lower() for c in df.columns]
    if any("ticker" in c for c in lower_cols):
        raise ValueError("ticker column detected")
    return True


def dataset_cleaning(data):
    df = data.copy()

    date_column = None
    for col in df.columns:
        converted = pd.to_datetime(df[col], errors='coerce', format='mixed')
        if converted.notna().sum() > len(df) * 0.5:
            df[col] = converted
            date_column = col
            break

    for col in df.columns:
        if col == date_column:
            continue
        converted = pd.to_numeric(df[col], errors='coerce')
        if converted.isna().sum() > df[col].isna().sum():
            df[col] = pd.to_numeric(
                df[col].astype(str).str.replace(r'[^0-9.-]', '', regex=True),
                errors='coerce',
            )
        else:
            df[col] = converted

    if date_column is not None:
        df = df.dropna(subset=[date_column])
        if not df[date_column].is_monotonic_increasing:
            df = df.sort_values(by=date_column).reset_index(drop=True)
        month = df[date_column].dt.month
        df['month_sin'] = np.sin(2 * np.pi * month / 12)
        df['month_cos'] = np.cos(2 * np.pi * month / 12)

    if df.isna().sum().sum() > 0:
        df = df.interpolate(method='linear').ffill().dropna()

    return df


def check_dataset_sufficiency(df, N, K):
    if len(df) < (N + K):
        new_rows = []
        while len(df) + len(new_rows) < (N + K):
            rows = np.random.normal(
                loc=df.mean(numeric_only=True),
                scale=df.std(numeric_only=True) * 0.05
            )
            new_rows.append(rows)
        new_df = pd.DataFrame(new_rows, columns=df.select_dtypes(include=[np.number]).columns)
        df = pd.concat([df, new_df], ignore_index=True)
    return df


def sliding_window(df, N, K):
    X, y = [], []
    for i in range(len(df) - N - K + 1):
        X.append(df[i: i + N])
        y.append(df[i + N: i + N + K])
    return np.array(X), np.array(y)


def data_preparation(df, N, K):
    split = int(len(df) * 0.8)
    X_train, y_train = sliding_window(df[:split], N, K)
    X_test, y_test = sliding_window(df[split:], N, K)

    scaler_X = MinMaxScaler(feature_range=(0, 1))
    scaler_y = MinMaxScaler(feature_range=(0, 1))

    X_train = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    y_train = scaler_y.fit_transform(y_train.reshape(-1, y_train.shape[-1])).reshape(y_train.shape)
    X_test = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    y_test = scaler_y.transform(y_test.reshape(-1, y_test.shape[-1])).reshape(y_test.shape)

    return X_train, X_test, y_train, y_test


class PredictiveModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, K, n_heads):
        super().__init__()
        self.K = K
        self.output_dim = output_dim
        self.conv1d = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=n_heads, batch_first=True)
        self.fc = nn.Linear(hidden_dim, K * output_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1d(x))
        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        out = self.fc(attn_out[:, -1, :])
        return out.view(-1, self.K, self.output_dim)


def train_model(X_train, y_train, input_dim, hidden_dim, output_dim, K, n_heads, batch_size=32, epochs=50):
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)

    model = PredictiveModel(input_dim, hidden_dim, output_dim, K, n_heads)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for _ in range(epochs):
        model.train()
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            loss = criterion(model(batch_X), batch_y)
            loss.backward()
            optimizer.step()
    return model


def run_training(df, N, K):
    df = check_dataset_sufficiency(df, N, K)
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True

    dataframes = df.copy()
    values = df.select_dtypes(include=[np.number]).values
    if values.ndim == 1:
        values = values.reshape(-1, 1)

    X_train, X_test, y_train, y_test = data_preparation(values, N, K)
    input_dim = X_train.shape[-1]
    hidden_dim = input_dim * 8
    output_dim = y_train.shape[-1]
    n_heads = 8 if hidden_dim % 8 == 0 else 4

    model = train_model(X_train, y_train, input_dim, hidden_dim, output_dim, K, n_heads)

    X_test_t = torch.tensor(X_test, dtype=torch.float32, requires_grad=True)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_test_t, y_test_t), batch_size=32, shuffle=False)

    model.eval()
    preds = []
    with torch.no_grad():
        for batch_X, _ in loader:
            preds.append(model(batch_X).numpy())
    final_predictions = np.concatenate(preds, axis=0)

    return model, final_predictions, X_test_t, dataframes


def captum_explainability(model, X_test_t, dataframes):
    model.eval()
    baseline = torch.zeros_like(X_test_t)
    ig = IntegratedGradients(model)
    attributions, _ = ig.attribute(
        inputs=X_test_t, baselines=baseline, target=(0, 0), return_convergence_delta=True
    )
    attr_np = attributions.detach().numpy()
    mean_attr = np.mean(np.abs(attr_np), axis=(0, 1))
    feature_importance = np.atleast_1d(np.mean(mean_attr, axis=0))

    columns = dataframes.select_dtypes(include=[np.number]).columns.tolist()
    importance_map = {name: float(score) for name, score in zip(columns, feature_importance)}
    total = sum(importance_map.values()) or 1
    return {k: round((v / total) * 100, 2) for k, v in importance_map.items()}


def generate_llm_explication(importance_map, final_predictions, data, api_key, model_name=DEFAULT_GROQ_MODEL):
    client = Groq(api_key=api_key)
    context_rows = data.tail(3).to_dict(orient='records')

    prompt = f"""[SYSTEM] You are an elite AI assistant specialized in Explainable AI (XAI) and Data Synthesis.
    Your job is to translate mathematical feature weights, predictions, and raw historical data into a clear summary.
    Stick strictly to the facts. Never hallucinate variables, trends, or domain-specific theories.
    [CONTEXT - USER RAW MARKET DATA (LAST 3 RECORDS)]
    {context_rows}
    [INFERENCE - MODEL PREDICTIONS]
    {final_predictions}
    [EXPLANATION - CAPTUM FEATURE ATTRIBUTIONS]
    {importance_map}
    [INSTRUCTION]
    Write a concise analysis for the user that MUST be formatted as ONE SINGLE CONTINUOUS PARAGRAPH. Do NOT use multiple lines, bullet points, numbered lists, or line breaks.
    In this paragraph, you must:
    1. Indicate whether the model predicts an upward or downward trend for the asset, using the latest actual prices from the raw market data as a starting point, while linking the data to what is stated in the raw market data, without providing the scaled values but the actual values from the table.
    2. Explain precisely how and why the specific characteristics determined this choice according to Captum's mathematical weightings, linking this behavior to market factors.
    3. Justify the reliability of this prediction by linking the historical benchmarks of actual prices, derived from the raw data, to the mathematical proofs.
    [EXPLANATION]:"""

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=500,
    )
    full_text = response.choices[0].message.content or ""

    if "</think>" in full_text:
        return full_text.split("</think>")[-1].strip()
    if "[EXPLANATION]:" in full_text:
        return full_text.split("[EXPLANATION]:")[-1].strip()
    return full_text.strip()


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.warning(
    "**Before you upload — please read:**\n"
    "- 📊 **The richer the dataset, the better the prediction.** More numeric "
    "columns and more history give the model more signal.\n"
    "- 📅 **Adding a date column as the first column is recommended.**\n"
    "- 🚫 **A `ticker` column will be rejected** (mixes multiple companies).\n"
    "- 🏢 **The dataset must contain data for ONE company only.**"
)

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

col1, col2 = st.columns(2)
with col1:
    N = st.number_input("History window size (N)", min_value=1, value=10, step=1)
with col2:
    K = st.number_input("Prediction horizon (K)", min_value=1, value=3, step=1)

groq_key_input = st.text_input(
    "Groq API key (optional if GROQ_API_KEY is already set as an env var / secret)",
    type="password",
)

if st.button("🚀 Run analysis", type="primary") and uploaded_file is not None:
    try:
        with st.spinner("Running the full pipeline... this may take a few minutes."):
            raw_data = pd.read_csv(uploaded_file)
            validate_dataset(raw_data)
            clean_df = dataset_cleaning(raw_data)
            model, predictions, X_test_t, dataframes = run_training(clean_df, int(N), int(K))
            importance_map = captum_explainability(model, X_test_t, dataframes)

            api_key = groq_key_input or os.environ.get("GROQ_API_KEY")
            if not api_key:
                try:
                    api_key = st.secrets.get("GROQ_API_KEY")
                except Exception:
                    api_key = None
            if not api_key:
                raise RuntimeError("missing Groq API key")

            explanation = generate_llm_explication(importance_map, predictions, dataframes, api_key)

        # --- Result: chart built from the Captum dictionary, text right below ---
        st.header("Result")

        imp_df = pd.DataFrame(
            list(importance_map.items()), columns=["Variable", "Importance (%)"]
        ).sort_values("Importance (%)", ascending=False)

        fig, ax = plt.subplots()
        ax.bar(imp_df["Variable"], imp_df["Importance (%)"])
        ax.set_ylabel("Importance (%)")
        ax.set_xticklabels(imp_df["Variable"], rotation=45, ha="right")
        st.pyplot(fig)

        st.write(explanation)

    except Exception:
        st.error(
            "😔 Sorry, something went wrong while processing your request. "
            "Your dataset appears to be inappropriate for this pipeline "
            "(wrong format, a mixed-company/ticker file, not enough usable data, "
            "or a missing Groq API key). Please check the requirements above and try again."
        )
elif uploaded_file is None:
    st.info("Upload a CSV file to get started.")
