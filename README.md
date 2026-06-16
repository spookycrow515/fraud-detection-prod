# Credit Card Fraud Detection System

An interactive machine learning pipeline and analytics dashboard built with **Streamlit** and **Scikit-Learn** to identify, analyze, and flag fraudulent credit card transactions. 

The application demonstrates real-time decision threshold optimization, supports interactive manual or randomized transaction profiling, and provides side-by-side performance evaluation matrices across different dataset class imbalance rectification strategies (**None**, **class_weight**, and **SMOTE**).

---

## 🚀 Key Features

* **Live Fraud Verification Sandbox:** Evaluate standalone transactions by manually setting inputs or instantly injecting highly realistic transaction vectors from a pre-calculated reference pool.
* **Dynamic Decision Threshold Optimization:** Adjust the probability cutoff slider on the fly to see immediate, cascading changes to classification **Precision** and **Recall** live validation metrics.
* **Balanced Strategy Performance Matrix:** Compare three pipeline balancing approaches side-by-side using an automated comparison table and an F1-Score delta bar chart.
* **Resilient Dual-Mode Execution Configuration:**
  * **Active Training Mode (Dataset Present):** When `creditcard_2023.csv` is detected, the system executes complete data diagnostic cleaning loops, compiles an offline sample pool (`temp.csv`), trains all three balancing techniques sequentially, and saves separate localized model binaries (`.pkl`).
  * **Cached Deployment Mode (Dataset Missing):** If the master data source is detached, the application gracefully skips heavy re-training. It boots instantly using the serialized pre-trained model files and routes the random testing widgets directly through the lightweight `temp.csv` backup engine.

---

## 📊 Dataset & Reference Pool Details

By default, the application runs on **`creditcard_2023.csv`**. 

| Column | Description |
|--------|-------------|
| `id` | Row identifier (dropped during cleaning) |
| `V1`–`V28` | Anonymized PCA-transformed features |
| `Amount` | Transaction currency amount |
| `Class` | Target label: `0` = Genuine/Legit, `1` = Fraudulent |

### Stratified Fallback Data (`temp.csv`)
When the master data is first processed, the model pipeline automatically extracts a 1,000-row stratified sub-sample written directly to `temp.csv`. To mirror real-world corporate fraud distribution anomalies while maintaining an active testing selection pool, this file is hardcoded to adhere to an exact constraint profile:
* **90% Genuine Cases** (900 baseline rows)
* **10% Fraud Cases** (100 verification rows)

---

## Managing Structural Imbalance: The Three Strategies

Because real-world financial data is severely skewed (often 99.9% genuine vs. 0.1% fraud), models require explicit algorithmic guidance to avoid blindly guessing "genuine" every time. This dashboard explores three built-in paradigms:

1. **None:** The baseline model trains directly on the default input distribution with standard uniform loss coefficients.
2. **Class Weight (`class_weight="balanced"`):** Keeps the original dataset intact but adjusts the mathematical optimization penalty. It assigns higher misclassification costs to minority labels (Fraud) inversely proportional to their class frequency.
3. **SMOTE (Synthetic Minority Over-sampling Technique):** Alters the physical training data matrix. It analyzes minority class feature neighborhoods via $k$-Nearest Neighbors ($k$-NN) and injects novel synthetic fraud vectors between them to balance the layout.

*Note: Since pre-cleaned public reference assets like `creditcard_2023.csv` are already balanced 50/50 by their creators, toggling these strategies on this specific data yields matching evaluation metrics down to the decimal point because structural skew correction triggers are naturally zeroed out.*

---

## 🛠️ Project Structure

```text
internship/
├── app.py                         # UI layer, navigation layout grids, and dashboard view loops
├── model.py                       # Pipeline processing, sampling constraints, and training orchestrator
├── temp.csv                       # Generated 90/10 stratified dataset for offline testing mock-ups
├── fraud_model_none.pkl           # Pre-trained model using raw baseline weights
├── fraud_model_class_weight.pkl  # Pre-trained model using balanced structural class weights
├── fraud_model_smote.pkl          # Pre-trained model using synthetic oversampling matrices
└── requirements.txt               # Manifest pinning required execution package versions
