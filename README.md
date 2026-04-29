[🇧🇷] [Leia em Português](README.pt.md)

# 🧠 Fake News Classification System (PT-BR)

This repository contains a machine learning model for detecting fake news in Portuguese, using the datasets [Fake.br-Corpus](https://github.com/roneysco/Fake.br-Corpus), [FakeTrue.br](https://github.com/Gabriel-Lino-Garcia/FakeRecogna) and [FakeRecogna](https://github.com/jpchav98/FakeTrue.Br).

The solution evolves from classic LSTM approaches to the use of Transformer-based models, such as BERT, allowing for a better capture of the news' semantic context.

---

## 🤗 Model on Hugging Face

The trained model is publicly available on the Hugging Face Hub:

👉 [https://huggingface.co/ericshantos/veritas-bert-ptbr/](https://huggingface.co/ericshantos/veritas-bert-ptbr/)

---

## 🚀 Objective

To develop a system capable of automatically classifying news as **true** or **false**, assisting in the fight against misinformation in the Portuguese language.

---

## 🧪 Technologies Used

* Python
* Pandas
* PyTorch
* Scikit-learn
* Jupyter Notebook
* Hugging Face Transformers (BERT)

---

## 📂 Dataset (Data Expansion)
The current version of the project utilizes a consolidated base from three major sources, tripling the original data volume to ensure greater generalization power:

| Source | Description |
| :--- | :--- |
| **Fake.br-Corpus** | Reference dataset with real and fake news. |
| **FakeTrue.br** | Complementary Portuguese news database. |
| **FakeRecogna** | Expanded dataset for greater thematic diversity. |

* **Total Volume:** ~22,684 news items (previously ~7,000).
* **Distribution:** 90% training / 10% testing with stratified sampling.

---

## 🧠 Model Architecture (BERT)
The model uses **BERTimbau** (BERT base for Portuguese) as its backbone, with a custom classification head:

* **Encoder:** `neuralmind/bert-base-portuguese-cased`.
* **Classification Head:**
    * Linear (Hidden Size → 32) + GELU Activation.
    * Dropout (0.2) for regularization.
    * Linear (32 → 16) + GELU Activation.
    * Linear (16 → 1) for binary output.
* **Optimization:** Adam with a Learning Rate of $5e^{-5}$.

## ⚙️ Data Pipeline
Processing now features specific extractors for each base (`BaseExtractor`):
1.  **Extraction:** Parsing `.txt` (Fake.br), `.csv` (FakeTrue), and `.xlsx` (FakeRecogna) files.
2.  **Cleaning:** Removal of null values and label normalization.
3.  **Tokenization:** WordPiece (BERT) with `max_length=256`.
4.  **Dataloader:** Implementation with `pin_memory` and `prefetch_factor` for GPU optimization.

---

## ⚙️ Training

### 📌 Hyperparameters (LSTM)

* Epochs: 5
* Batch size: 128
* Optimizer: Adam
* Loss: Binary Crossentropy

### 📌 BERT (Fine-tuning)

* Learning rate: ~2e-5
* Batch size: 32
* GPU usage recommended

---

## 📊 Results

The LSTM model achieved approximately **98% accuracy** on the test set.

> BERT-based models show significant potential for improvement by capturing linguistic context more effectively.

![Result](./assets/result.png)

---

## 🚀 How to Use the Model

The model can be loaded directly via Transformers:

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "neuralmind/bert-base-portuguese-cased"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=2  
)

state_dict = torch.load("veritas-bert-ptbr.pth", map_location=device)

model.load_state_dict(state_dict)

model.to(device)
model.eval()
```

---


## ▶️ How to Run

1. Clone the repository:

```bash
git clone https://github.com/ericshantos/veritas_br.git
```

2. Run the notebook:

* [Run on Google Colab](https://colab.research.google.com/github/ericshantos/br_fake_news_detector_model/blob/main/br_fake_news_detector_model.ipynb)
* Or locally:

```bash
jupyter notebook veritas_br.ipynb
```

---

## 💡 Project Insights

* LSTM models are efficient but semantically limited.
* BERT significantly improves context understanding.
* Tokenization is a critical factor for performance.

---

## 💐 Acknowledgments

I dedicate this project to my high school teachers, who contributed to the development of my critical thinking.

Special mention to Professor Winola Cunha, who reinforced the importance of morphosyntax — and was absolutely right.

---

## 📜 License

This project is under the MIT license. See [LICENSE](./LICENSE) for more details.

---

**Created by Eric dos Santos 🚀**