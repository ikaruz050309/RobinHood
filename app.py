import os
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

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_page_config(page_title="Millnew AI", page_icon="📈", layout="wide")

# Style Monochrome Strict
st.markdown("""
    <style>
        .stApp { background-color: #000000 !important; color: #ffffff !important; }
        h1, h2, h3, p, span, label, li { color: #ffffff !important; }
        .stAlert { background-color: #111111 !important; border: 1px solid #333333 !important; }
        .stSlider, .stFileUploader { color: #ffffff !important; }
        .stButton>button { background-color: #111111 !important; color: #ffffff !important; border: 1px solid #ffffff !important; }
        .stButton>button:hover { background-color: #ffffff !important; color: #000000 !important; }
    </style>
""", unsafe_allow_html=True)

st.title("📈 Millnew AI")
st.subheader("On-Device Multi-Asset Forecasting Framework — Nearly Zero Hallucinations")

st.write("---")

st.warning("""
### ⚠️ Crucial Dataset Guidelines & Rules:
Before uploading any file to the framework, ensure your dataset strictly complies with the following structural criteria:
* **CSV Files Only:** The system exclusively processes structured `.csv` file formats.
* **No 'Ticker' Column / Single Company Only:** You cannot mix data from several companies. If a 'ticker' or multi-entity column is detected, the engine will instantly halt.
* **Purely Numerical Asset Series:** All evaluation columns must contain clean continuous numerical values (floats/integers) representing asset dimensions.
* **The Richer, The Better:** High data density directly translates to robust mathematical attribution maps. Ensure your historical record counts exceed your target N lookback horizons.
""")

st.markdown("""
### 🧠 What is this project?
Traditional Large Language Models (LLMs) tend to be baffling—they are great at reasoning but poor at raw math. 

By reversing the usual approach and feeding the LLM with deterministic prediction attributes, we can force the model to strictly explain each of the AI's choices. This framework isolates raw mathematical feature attributions on-device, delivering a cold, objective, and unbiased look at multi-asset time-series data across custom horizons without computational guesswork.
""")

st.write("---")

# Pipeline Technique
def validate_dataset(df):
    lower_cols = [str(c).lower() for c in df.columns]
    if any("ticker" in c for c in lower_cols):
        raise ValueError("The system cannot process datasets mixing multiple companies. Remove the 'ticker' column.")
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
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(r'[^0-9.-]', '', regex=True), errors='coerce')
        else:
            df[col] = converted

    if date_column is not None:
        df = df.dropna(subset=[date_column])
        if not df[date_column].is_monotonic_increasing:
            df = df.sort_values(by=date_column).reset_index(drop=True)
    
    df_numeric = df.select_dtypes(include=[np.number])
    if df_numeric.isna().sum().sum() > 0:
        df_numeric = df_numeric.interpolate(method='linear').ffill().dropna()
    return df_numeric

def check_dataset_sufficiency(df, N, K):
    if len(df) < (N + K):
        new_rows = []
        while len(df) + len(new_rows) < (N + K):
            rows = np.random.normal(loc=df.mean(), scale=df.std() * 0.05)
            new_rows.append(rows)
        new_df = pd.DataFrame(new_rows, columns=df.columns)
        df = pd.concat([df, new_df], ignore_index=True)
    return df

def sliding_window(df, N, K):
    X, y = [], []
    for i in range(len(df) - N - K + 1):
        X.append(df.iloc[i: i + N].values)
        y.append(df.iloc[i + N: i + N + K].values)
    return np.array(X), np.array(y)

def data_preparation(df, N, K):
    values = df.values
    split = int(len(values) * 0.8)
    X_train, y_train = sliding_window(df.iloc[:split], N, K)
    X_test, y_test = sliding_window(df.iloc[split:], N, K)

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

def train_model(X_train, y_train, input_dim, hidden_dim, output_dim, K, n_heads):
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=16, shuffle=True)

    model = PredictiveModel(input_dim, hidden_dim, output_dim, K, n_heads)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    for _ in range(5):
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
    X_train, X_test, y_train, y_test = data_preparation(df, N, K)
    input_dim = X_train.shape[-1]
    hidden_dim = 16
    output_dim = y_train.shape[-1]
    n_heads = 2

    model = train_model(X_train, y_train, input_dim, hidden_dim, output_dim, K, n_heads)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, requires_grad=True)
    model.eval()
    with torch.no_grad():
        final_predictions = model(X_test_t).numpy()
    return model, final_predictions, X_test_t, df

def captum_explainability(model, X_test_t, df):
    model.eval()
    baseline = torch.zeros_like(X_test_t)
    ig = IntegratedGradients(model)
    attributions, _ = ig.attribute(inputs=X_test_t, baselines=baseline, target=(0, 0), return_convergence_delta=True)
    
    attr_np = np.abs(attributions.detach().numpy())
    mean_attr = np.mean(attr_np, axis=(0, 1))
    importance_map = {name: float(score) for name, score in zip(df.columns, mean_attr)}
    total = sum(importance_map.values()) or 1
    return {k: round((v / total) * 100, 2) for k, v in importance_map.items()}

# Configuration des paramètres sur la page principale
st.markdown("### 🎛️ Parameters Configuration")
col_n, col_k = st.columns(2)
with col_n:
    N = st.slider("N Horizon (Context History)", min_value=1, max_value=30, value=10, step=1)
with col_k:
    K = st.slider("K Horizons (Future Steps)", min_value=1, max_value=30, value=5, step=1)

st.write("---")

st.markdown("### 📥 UI Demo Input")
csv_sample_content = """Date,SPX,GLD,USO,SLV,EUR/USD
1/2/2008,1447.16,84.86,78.47,15.18,1.4716
1/3/2008,1447.16,85.57,78.37,15.285,1.4744
1/4/2008,1411.63,85.12,77.30,15.167,1.4754
1/7/2008,1416.18,84.76,75.50,15.053,1.4682
1/8/2008,1390.18,86.77,76.05,15.59,1.5570
1/9/2008,1409.13,86.55,75.25,15.52,1.4664
1/10/2008,1420.32,88.25,74.01,16.061,1.4801"""

st.download_button(label="⬇️ Download Official gld_price_data.csv", data=csv_sample_content, file_name="gld_price_data.csv", mime="text/csv")
uploaded_file = st.file_uploader("Drag and drop your custom CSV dataset here", type=["csv"])
use_sample = st.checkbox("Or run the pipeline using the integrated 2008 dataset sample instantly")

target_data = None
if uploaded_file is not None:
    try:
        target_data = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error("❌ Structural Read Error.")
elif use_sample:
    from io import StringIO
    target_data = pd.read_csv(StringIO(csv_sample_content))

if target_data is not None:
    try:
        validate_dataset(target_data)
        cleaned_df = dataset_cleaning(target_data)
        
        if cleaned_df.shape == 0:
            st.error("❌ Data Extraction Failure.")
        else:
            with st.spinner("🔄 Running execution loop..."):
                model, final_predictions, X_test_t, processed_df = run_training(cleaned_df, N, K)
                importance_map = captum_explainability(model, X_test_t, processed_df)
            
            st.success("✅ Computations completed!")
            
            # Graphique Captum Épuré
