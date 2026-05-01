[🇺🇸] [Read in English](README.md)

# 🧠 Sistema de Classificação de Fake News (PT-BR)

Este repositório contém um modelo de aprendizado de máquina para detecção de fake news em português, utilizando os datasets [Fake.br-Corpus](https://github.com/roneysco/Fake.br-Corpus), [FakeTrue.br](https://github.com/jpchav98/FakeTrue.Br) e [FakeRecogna](https://github.com/Gabriel-Lino-Garcia/FakeRecogna).

A solução evolui de abordagens clássicas com LSTM para o uso de modelos baseados em Transformers, como o BERT, permitindo uma melhor captura do contexto semântico das notícias.

---

## 🤗 Modelo no Hugging Face

O modelo treinado está disponível publicamente no Hugging Face Hub:

👉 https://huggingface.co/ericshantos/veritas-bert-ptbr/

---

## 🚀 Objetivo

Desenvolver um sistema capaz de classificar automaticamente notícias como **verdadeiras** ou **falsas**, auxiliando no combate à desinformação na língua portuguesa.

---

## 🧪 Tecnologias Utilizadas

* Python  
* Pandas  
* PyTorch  
* Scikit-learn  
* Jupyter Notebook  
* Hugging Face Transformers (BERT)

---

## 📂 Dataset (Expansão de Dados)

A versão atual do projeto utiliza uma base consolidada de três grandes fontes, triplicando o volume original de dados para garantir maior capacidade de generalização:

| Fonte | Descrição |
| :--- | :--- |
| **Fake.br-Corpus** | Dataset de referência com notícias reais e falsas |
| **FakeTrue.br** | Base complementar de notícias em português |
| **FakeRecogna** | Dataset expandido com maior diversidade temática |

* **Volume total:** ~22.684 notícias (antes ~7.000)  
* **Divisão:** 90% treino / 10% teste com amostragem estratificada  

---

## 🧠 Arquitetura do Modelo (BERT)

O modelo utiliza o **BERTimbau** (BERT base para português) como backbone, com uma cabeça de classificação personalizada:

* **Encoder:** `neuralmind/bert-base-portuguese-cased`
* **Cabeça de Classificação:**
  * Linear (Hidden Size → 32) + GELU  
  * Dropout (0.2) para regularização  
  * Linear (32 → 16) + GELU  
  * Linear (16 → 2) para saída binária  
* **Função de perda:** CrossEntropyLoss  
* **Otimização:** Adam com learning rate de ~5e-5  

---

## ⚙️ Pipeline de Dados

O processamento utiliza extratores específicos para cada base (`BaseExtractor`):

1. **Extração:** leitura de arquivos `.txt` (Fake.br), `.csv` (FakeTrue) e `.xlsx` (FakeRecogna)  
2. **Limpeza:** remoção de valores nulos e normalização dos rótulos  
3. **Tokenização:** WordPiece (BERT) com `max_length=256`  
4. **Dataloader:** uso de `pin_memory` e `prefetch_factor` para otimização em GPU  

---

## ⚙️ Treinamento

### 📌 LSTM (versões anteriores)

* Épocas: 5  
* Batch size: 128  
* Otimizador: Adam  
* Loss: Binary Crossentropy  

### 📌 BERT (Fine-tuning)

* Learning rate: ~2e-5 a 5e-5  
* Batch size: 32  
* Uso de GPU recomendado  

---

## 📊 Resultados

O modelo alcançou aproximadamente:

* **Acurácia:** ~97.5%  
* **F1-score:** ~97.5%  
* **Alta precisão e recall balanceados**

> Modelos baseados em BERT demonstram grande capacidade de capturar contexto linguístico, superando abordagens tradicionais.

![Resultado](./assets/result.png)

---

## 🚀 Como Usar o Modelo

O modelo pode ser carregado diretamente com Transformers:

```python
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

repo_id = "ericshantos/veritas-bert-ptbr"

tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForSequenceClassification.from_pretrained(repo_id)

model.to(device)
model.eval()
````

---

## ▶️ Como Executar

1. Clone o repositório:

```bash
git clone https://github.com/ericshantos/veritasBR.git
```

2. Execute o notebook:

* Google Colab:
  [https://colab.research.google.com/github/ericshantos/veritasBR/blob/main/veritasBR_v3.ipynb](https://colab.research.google.com/github/ericshantos/veritasBR/blob/main/veritasBR_v3.ipynb)

* Ou localmente:

```bash
jupyter notebook veritasBR.ipynb
```

---

## 💡 Insights do Projeto

* Modelos LSTM são eficientes, porém limitados semanticamente
* BERT melhora significativamente a compreensão de contexto
* A tokenização é um fator crítico para o desempenho
* A qualidade e diversidade dos dados impactam diretamente os resultados

---

## 💐 Agradecimentos

Dedico este projeto aos meus professores do ensino médio, que contribuíram para o desenvolvimento do meu pensamento crítico.

Menção especial à professora Winola Cunha, que reforçou a importância da morfossintaxe — e estava absolutamente certa.

---

## 📜 Licença

Este projeto está sob a licença MIT. Consulte o arquivo [LICENSE](./LICENSE) para mais detalhes.

---

**Criado por Eric dos Santos 🚀**
