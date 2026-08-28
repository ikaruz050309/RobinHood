# RobinHood

An open-source, hybrid local framework designed to analyze multi-asset time-series data. It leverages a deep learning architecture (TCN + LSTM + Attention) combined with a mathematical feature attribution layer (Captum Integrated Gradients) to eliminate hallucinations in automated financial reports.

> 📊 **Quick Start:** A ready-to-use `gld_price_data.csv` dataset is included in the repository so you can test the framework instantly.

---

## 🧠 Why This Project?

Traditional Large Language Models (LLMs) tend to be baffling—they are great at reasoning but poor at raw math. When processing raw numerical tables or dense time-series, autoregressive models often invent trends or hallucinate metrics, breaking structural analytical trust.

By reversing the usual approach and feeding the language interface with deterministic prediction attributes, we can force the model to strictly explain each of the AI's choices. 

This framework isolates raw mathematical feature attributions **on-device**, delivering a cold, objective, and unbiased look at multi-asset data across custom horizons without computational guesswork or statistical fabrication.

---

## 🛠️ Architecture & Privacy Design

1. **Local Compute (100% On-Device):** Your raw time-series data, deep learning training loops (TCN + LSTM), and Captum mathematical feature attributions run entirely locally on your hardware.
2. **Text Synthesis (Hybrid Cloud):** Only the final, anonymized deterministic numerical weights are sent via API to open-source LLMs on Hugging Face to generate the human-readable text report. **Your raw data never leaves your machine.**

---

## 🚀 How to Use & Setup

Follow these precise steps to deploy and execute the pipeline locally on your machine:

### 1. Clone the Architecture
```bash
gh repo clone ikaruz050309/RobinHood
cd RobinHood
```

### 2. Install Environment Dependencies
Ensure you have Python installed, then run the installation command to fetch all background packages:
```bash
pip install -r requirements.txt
```

### 3. Configure Your Environment Keys
To enable natural language synthesis from your local evaluation weights via Hugging Face Serverless Inference, export your API token:
```bash
export HUGGINGFACE_API_KEY="your_huggingface_api_key_here"
```

### 4. Launch the Framework
```bash
python3 run main.py
```
---

## ⚠️ Dataset Guidelines & Compliance Rules

If you want to upload your own custom data, your `.csv` dataset must match the following technical parameters to prevent matrix dimensions or compliance checks from halting the pipeline:

* **CSV Format Only:** The system exclusively parses structured `.csv` files.
* **Single Company Constraint:** The architecture evaluates standalone data matrices. Do not include a 'Ticker' column or mix data from multiple corporations, or the engine will immediately halt the execution loop.
* **Purely Numerical Time-Series:** Except for a single chronologically sequential date column, all tracking columns must contain pure continuous float or integer sequences.
* **Data Density (The Richer, The Better):** High data density directly translates to robust mathematical attribution maps. Ensure your historical record counts vastly exceed your configured context windows ($N$) and prediction horizons ($K$).

