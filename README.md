# 🤖 A Comparative Study of Transformer Models for Sarcasm Detection

This repository presents a comparative study of three transformer-based models—**DistilBERT**, **MiniLM**, and **ELECTRA**—for sarcasm detection using the [Kaggle News Headlines Sarcasm Dataset](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection).

## 📘 Project Summary

Sarcasm detection is a challenging task in Natural Language Understanding (NLU) due to the gap between literal and intended meanings. This project explores how well different transformer models understand and classify sarcastic content in news headlines by fine-tuning and evaluating their performance.

---

## 📂 Dataset

- **Source**: Kaggle – *News Headlines Dataset for Sarcasm Detection*
- **Size**: ~28,619 news headlines
- **Labels**:
  - `1`: Sarcastic
  - `0`: Non-sarcastic
- **Train/Test Split**: 80/20 with balanced class distribution
- **Columns Used**: `headline` (renamed to `text`), `is_sarcastic` (renamed to `label`)

---

## 🛠 Models Used

| Model       | Description                                                                 |
|-------------|-----------------------------------------------------------------------------|
| DistilBERT  | A smaller, faster version of BERT retaining ~97% of its performance         |
| MiniLM      | Lightweight model distilling self-attention from deeper models             |
| ELECTRA     | Efficient pretraining with replaced token detection instead of masking     |

---

## ⚙️ Methodology

1. **Preprocessing**: Tokenization using model-specific tokenizers, padding, truncation.
2. **Fine-Tuning**: Each model was trained using Hugging Face Transformers on the same hyperparameters.
3. **Evaluation**: Accuracy, F1-score, Precision, Recall, Confusion Matrix.

---

## 📊 Results

| Model       | Accuracy | F1-Score | Precision | Recall |
|-------------|----------|----------|-----------|--------|
| MiniLM      | **94%**  | **0.94** | 0.94      | 0.94   |
| DistilBERT  | 93%      | 0.93     | 0.93      | 0.93   |
| ELECTRA     | 92%      | 0.92     | 0.93      | 0.92   |

- **MiniLM** demonstrated the most balanced and effective performance across all metrics.
- **DistilBERT** offered a good trade-off between speed and performance.
- **ELECTRA** performed conservatively, slightly underpredicting sarcasm.

---

## 🧪 Tech Stack

- Python
- Hugging Face Transformers
- Scikit-learn
- PyTorch
- Pandas / NumPy / Matplotlib

---

## 📈 Visualizations

- Confusion Matrices for each model
- Accuracy and F1-score comparison table

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/yourusername/sarcasm-transformer-comparison.git
cd sarcasm-transformer-comparison

# (Optional) Create a virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook Model_Comparision.ipynb
