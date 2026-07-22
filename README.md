# 💳 Credit Card Fraud Detection using Machine Learning & Streamlit

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)

---

## 📌 Project Overview

This project is a **Credit Card Fraud Detection System** built using **Python, Machine Learning, and Streamlit**. It detects fraudulent credit card transactions by training multiple machine learning models on an imbalanced dataset and automatically selecting the best-performing model based on evaluation metrics.

The application provides an interactive **Streamlit dashboard** that allows users to:

- Explore the dataset
- Visualize fraud patterns
- Upload transaction CSV files
- Predict fraudulent transactions instantly
- Download prediction results

The trained model is saved as a **`.pkl`** file, ensuring that predictions are performed without retraining the model.

---

## 🚀 Features

- ✅ End-to-End Machine Learning Pipeline
- ✅ Automatic Data Preprocessing
- ✅ SMOTE for Class Imbalance Handling
- ✅ Multiple Machine Learning Algorithms
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)
- ✅ Automatic Best Model Selection
- ✅ Model Serialization using Joblib (`.pkl`)
- ✅ Interactive Streamlit Dashboard
- ✅ CSV Upload & Batch Prediction
- ✅ Download Prediction Results
- ✅ Professional Data Visualizations
- ✅ Modular Project Structure
- ✅ Clean & Well-Documented Code

---

# 🛠️ Technologies Used

- Python 3.11+
- Streamlit
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Imbalanced-learn (SMOTE)
- Joblib

---

# 📂 Project Structure

```text
Credit-Card-Fraud-Detection/

│
├── dataset/
│   ├── creditcard.csv          # Download from Kaggle
│   └── sample_input.csv        # Generated during training
│
├── models/
│   ├── fraud_model.pkl
│   └── scaler.pkl
│
├── outputs/
│   ├── best_model_info.json
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   └── ...
│
├── app.py
├── train_model.py
├── utils.py
├── config.py
├── requirements.txt
├── README.md
└── LICENSE
```

---

# 📊 Dataset

This project uses the **Credit Card Fraud Detection Dataset** available on Kaggle.

**Dataset Link**

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

> **Note:**  
> The dataset is **not included** in this repository because it exceeds GitHub's upload size limit.

After downloading:

1. Extract the ZIP file.
2. Copy `creditcard.csv`.
3. Place it inside the `dataset/` folder.

Example:

```text
dataset/
    creditcard.csv
```

---

# ⚙️ Installation

## 1️⃣ Clone the Repository

```bash
git clone https://github.com/btr2006/Credit-Card-Fraud-Detection.git
```

```bash
cd Credit-Card-Fraud-Detection
```

---

## 2️⃣ Create a Virtual Environment (Recommended)

### Windows

```bash
python -m venv venv
```

```bash
venv\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv venv
```

```bash
source venv/bin/activate
```

---

## 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🤖 Model Training

Run the training script:

```bash
python train_model.py
```

The training pipeline will:

- Load the dataset
- Preprocess the data
- Apply SMOTE
- Train multiple ML models
- Compare performance
- Select the best model
- Save:

```text
models/
│
├── fraud_model.pkl
└── scaler.pkl
```

It also generates evaluation plots and performance reports inside the `outputs/` folder.

---

# 🌐 Run the Streamlit Application

After training is complete:

```bash
streamlit run app.py
```

The application will launch locally at:

```text
http://localhost:8501
```

---

# 📈 Machine Learning Workflow

```
Dataset
    │
    ▼
Data Preprocessing
    │
    ▼
Train-Test Split
    │
    ▼
Feature Scaling
    │
    ▼
SMOTE
    │
    ▼
Train Multiple Models
    │
    ▼
Evaluate Performance
    │
    ▼
Select Best Model
    │
    ▼
Save Model (.pkl)
    │
    ▼
Streamlit Prediction Dashboard
```

---

# 📊 Evaluation Metrics

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve

---

# 📤 Streamlit Features

The application includes:

- 🏠 Home
- 📊 Dataset Overview
- 📈 Visualizations
- 🤖 Model Information
- 📥 Download Sample CSV
- 📤 Upload CSV
- 🔍 Fraud Prediction
- 📋 Prediction Results
- 📥 Download Prediction CSV

---

# 📄 Sample Input

Users can download a sample CSV directly from the application.

The sample file contains:

```
Time
V1
V2
...
V28
Amount
```

Users can upload the edited CSV for batch fraud prediction.

---

# 🔮 Future Improvements

- Deep Learning Models (ANN, Autoencoders)
- XGBoost & LightGBM
- Explainable AI using SHAP/LIME
- REST API with FastAPI
- Docker Containerization
- CI/CD Pipeline
- Cloud Deployment (AWS, Azure, GCP)
- Streamlit Cloud Deployment

---

# 📜 License

This project is licensed under the **MIT License**.

---

# 👨‍💻 Author

## **Bhanu Teja Reddy**

Computer Science Engineering (Artificial Intelligence & Machine Learning)

GitHub:
**https://github.com/btr2006**

Project Repository:
**https://github.com/btr2006/Credit-Card-Fraud-Detection**

---

## ⭐ Support

If you found this project useful:

⭐ Star this repository

🍴 Fork the repository

📢 Share it with others

Happy Coding! 🚀
