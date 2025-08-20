# 🔎 Malicious URL Detection

A **machine learning project in Python** for detecting **malicious URLs** (phishing, malware, spam, suspicious).  
The project implements **multiple classical ML and deep learning models**, ensemble methods, and detailed evaluation metrics.  

This repository is part of a larger research project on **cybersecurity and AI**.  

---

## ✨ Features
- 📊 **Data Preprocessing**
  - URL lexical feature extraction (length, number of dots, suspicious keywords, etc.)  
  - Domain-based features (TLD, presence of IP address, subdomain length)  
  - Special character statistics (`@`, `-`, `/`, `?`, `=`, `%`)  
  - Bag-of-words and TF-IDF for tokenized URL parts  
  - Selenium-based preprocessing for live data scraping  

- 🤖 **Machine Learning & Deep Learning Models**
  - **Classical ML:**
    - Random Forest (RFC)  
    - XGBoost (XGB)  
    - LightGBM (LGBM)  
    - TabNet (interpretable DL model for tabular data)  

  - **Deep Learning:**
    - DNN (3-layer and 5-layer architectures)  
    - Graph Neural Networks (GNN)  
    - Autoencoder-based classifiers (AE)  
    - RBFN (Radial Basis Function Network) + RFC  

  - **NLP-based:**
    - **BERT** for URL token classification  

  - **Ensemble:**
    - Stacking of base models with meta-classifier  
    - Model averaging and voting  

- 📈 **Evaluation**
  - Confusion Matrix, ROC curves  
  - Precision, Recall, F1-score  
  - Per-class analysis (Benign / Malware / Phishing / Spam)  

- 🖥️ **Menu-driven CLI** for experiments (`Menu.py`, `main.py`)  
- 📝 **Logging system** for reproducibility (`LogSystem.py`)  

---

## 📂 Project Structure
```
malicious_url_detection-master/
│── main.py                         # Entry point
│── Menu.py                         # Menu interface
│── DataPreprocessing.py            # Data preprocessing
│── TrainTestDataPreprocessing.py   # Train/Test split
│── TrainModels.py                  # Model training
│── CustomModels.py                 # DNN, GNN, AE, RBFN, etc.
│── Ensamble.py                     # Ensemble methods
│── Selenium_preprocessing.py       # URL preprocessing with Selenium
│── ConfusionMatrixAndRocCreator.py # Metrics visualization
│── LogSystem.py                    # Logging
```

---

## ⚙️ Requirements
- Python **3.8+**  
- Install dependencies:
```bash
pip install -r requirements.txt
```

If missing:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn selenium tensorflow keras lightgbm xgboost pytorch tabnet
```

---

## 🚀 Usage
### 1. Preprocess Data
```bash
python DataPreprocessing.py
python TrainTestDataPreprocessing.py
```

### 2. Train Models
```bash
python TrainModels.py
```

### 3. Evaluate Models
```bash
python ConfusionMatrixAndRocCreator.py
```

### 4. Run Menu App
```bash
python main.py
```

---

## 🔬 Example Extracted Features
| Feature | Description |
|---------|-------------|
| `url_length` | Total length of URL |
| `num_dots` | Number of dots in domain |
| `has_ip` | URL uses raw IP instead of domain |
| `tld` | Top-level domain (e.g., `.com`, `.ru`, `.xyz`) |
| `num_special_chars` | Count of `@`, `-`, `/`, `?`, `=` |
| `entropy` | Shannon entropy of URL string |
| `keyword_phish` | Presence of words like `login`, `secure`, `bank` |

---

## 📊 Results (example)
| Model  | Accuracy | Precision | Recall | F1-score |
|--------|----------|-----------|--------|----------|
| Random Forest | 0.93 | 0.92 | 0.91 | 0.91 |
| LightGBM | 0.95 | 0.94 | 0.93 | 0.94 |
| XGBoost | 0.96 | 0.95 | 0.94 | 0.95 |
| TabNet | 0.94 | 0.93 | 0.92 | 0.93 |
| DNN (5 layers) | 0.92 | 0.91 | 0.90 | 0.90 |
| GNN | 0.91 | 0.90 | 0.89 | 0.89 |
| Autoencoder Classifier | 0.90 | 0.88 | 0.87 | 0.88 |
| Stacking Ensemble | **0.97** | **0.96** | **0.96** | **0.96** |

---

## 🛠️ Development
- Language: **Python**  
- ML/DL Frameworks: **scikit-learn, TensorFlow, PyTorch, LightGBM, XGBoost**  
- NLP: **BERT**  
- Ensemble: **Stacking, Voting**  
- Visualization: **matplotlib, seaborn**  

---
