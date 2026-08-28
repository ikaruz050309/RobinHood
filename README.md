# RobinHood

An open-source, local framework designed to analyze multi-asset time-series data using a hybrid deep learning architecture (TCN + LSTM + Attention) and mathematical feature attribution layer (Captum Integrated Gradients) to eliminate hallucinations in automated reports.

---

## 🧠 Why This Project?

Traditional Large Language Models (LLMs) tend to be baffling—they are great at reasoning but poor at raw math. When processing raw numerical tables or dense time-series, autoregressive models often invent trends or hallucinate metrics, breaking structural analytical trust.

By reversing the usual approach and feeding the language interface with deterministic prediction attributes, we can force the model to strictly explain each of the AI's choices. This framework isolates raw mathematical feature attributions on-device, delivering a cold, objective, and unbiased look at multi-asset data across custom horizons without computational guesswork or statistical fabrication.

---

## 📂 Repository Structure

Ensure your local directory contains the following core files before execution:
* `main.py` — The core deep learning training loop and mathematical framework pipeline.
* `app.py` — The unified user interface layer.
* `requirements.txt` — The environment dependency manifest.

---

## 🚀 How to Use & Setup

Follow these precise steps to deploy and execute the pipeline locally on your machine:

### 1. Clone the Architecture
```bash
git clone https://github.com
cd YOUR_REPOSITORY
```

### 2. Install Environment Dependencies
Ensure you have Python installed, then run the installation command to fetch all background packages:
```bash
pip install -r requirements.txt
```

### 3. Configure Your Environment Keys
To enable natural language synthesis from your local evaluation weights without computing delays, make sure to export your Hugging Face API credential node:
```bash
export HUGGINGFACE_API_KEY="your_huggingface_api_key_here"
```

### 4. Launch the Framework
```bash
streamlit run app.py
```

---

## ⚠️ Dataset Guidelines & Compliance Rules

To prevent matrix dimensions or compliance checks from halting the pipeline, your custom `.csv` dataset must match the following technical parameters:

* **CSV Format Only:** The system exclusively parses structured `.csv` files.
* **Single Company Constraint:** The architecture evaluates standalone data matrices. **Do not include a 'Ticker' column** or mix data from multiple corporations, or the engine will immediately halt the execution loop.
* **Purely Numerical Time-Series:** Except for a single chronologically sequential date column, all tracking columns must contain pure continuous float or integer sequences.
* **Data Density (The Richer, The Better):** High data density directly translates to robust mathematical attribution maps. Ensure your historical record counts vastly exceed your configured context windows ($N$) and prediction horizons ($K$).

---
