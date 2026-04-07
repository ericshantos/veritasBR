[🇬🇧] [Read in English](README.md)

# 🧠 Sistema de Classificação de Notícias Falsas (PT-BR)

Este repositório contém um modelo de machine learning para detecção de fake news em português, utilizando o dataset [Fake.br-Corpus](https://github.com/roneysco/Fake.br-Corpus).

A solução evolui de abordagens clássicas com LSTM para o uso de modelos baseados em Transformers, como o BERT, permitindo capturar melhor o contexto semântico das notícias.

---

## 🤗 Modelo no Hugging Face

O modelo treinado está disponível publicamente no Hugging Face Hub:

👉 https://huggingface.co/ericshantos/veritas-bert-ptbr/

---

## 🚀 Objetivo

Desenvolver um sistema capaz de classificar automaticamente notícias como **verdadeiras** ou **falsas**, auxiliando no combate à desinformação em língua portuguesa.

---

## 🧪 Tecnologias Utilizadas

* Python
* Pandas
* PyTorch
* Scikit-learn
* Jupyter Notebook
* Hugging Face Transformers (BERT)

---

## 🧠 Arquitetura do Modelo

O projeto contempla duas abordagens principais:

### 🔹 Modelo 1 — LSTM (baseline)

* Camada de Embedding
* 3 camadas LSTM
* Dropout para regularização
* Camada densa com sigmoid

### 🔹 Modelo 2 — BERT (estado da arte)

* Modelo pré-treinado: `neuralmind/bert-base-portuguese-cased`
* Tokenização com WordPiece
* Fine-tuning para classificação binária
* Possibilidade de expansão de vocabulário com novos tokens

---

## 📂 Dataset

O dataset utilizado é o **Fake.br-Corpus**, contendo notícias reais e falsas em português.

### 📥 Download:

```bash
git clone https://github.com/roneysco/Fake.br-Corpus
```

Ou execute diretamente no notebook.

---

## 🗂️ Pipeline de Dados

O processamento inclui:

* Leitura e extração de textos
* Tokenização:

  * LSTM: tokenização clássica
  * BERT: WordPiece Tokenizer
* Padding e truncamento
* Split treino/teste (80/20)

---

## ⚙️ Treinamento

### 📌 Hiperparâmetros (LSTM)

* Épocas: 5
* Batch size: 128
* Otimizador: Adam
* Loss: Binary Crossentropy

### 📌 BERT (Fine-tuning)

* Learning rate: ~2e-5
* Batch size: 8–16
* Uso de GPU recomendado

---

## 📊 Resultados

O modelo LSTM atingiu aproximadamente **98% de acurácia** no conjunto de teste.

> Modelos baseados em BERT apresentam potencial de melhoria significativa ao capturar melhor o contexto linguístico.

![Resultado](./assets/result.png)

---

## 🚀 Como Utilizar o Modelo

O modelo pode ser carregado diretamente via Transformers:

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


## ▶️ Como Executar

1. Clone o repositório:

```bash
git clone https://github.com/ericshantos/veritas_br.git
```

2. Execute o notebook:

* [Executar no Google Colab](https://colab.research.google.com/github/ericshantos/br_fake_news_detector_model/blob/main/br_fake_news_detector_model.ipynb)
* Ou localmente:

```bash
jupyter notebook veritas_br.ipynb
```

---

## 💡 Insights do Projeto

* Modelos LSTM são eficientes, mas limitados semanticamente
* BERT melhora significativamente a compreensão de contexto
* Tokenização é um fator crítico para performance

---

## 💐 Agradecimentos

Dedico este projeto aos meus professores do Ensino Médio, que contribuíram para o desenvolvimento do meu senso crítico.

Menção especial à professora Winola Cunha, que reforçou a importância da morfossintaxe — e estava absolutamente certa.

---

## 📜 Licença

Este projeto está sob a licença MIT. Veja [LICENSE](./LICENSE) para mais detalhes.

---

**Criado por Eric dos Santos 🚀**
