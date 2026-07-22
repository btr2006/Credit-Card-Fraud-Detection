# Credit Card Fraud Detection using Machine Learning & Streamlit

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32.2-red.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4.1-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## Project Overview
A production-quality, modular, and interactive Credit Card Fraud Detection System built with Python and Streamlit. This project trains multiple machine learning models on highly imbalanced transaction data using SMOTE, automatically selects the best performing model based on ROC-AUC score, and serves real-time predictions through a user-friendly Streamlit web dashboard.

## Features
- **End-to-End ML Pipeline:** Data loading, preprocessing, scaling, SMOTE resampling, and model evaluation.
- **Multiple Models:** Trains Logistic Regression, Decision Tree, Random Forest, and SVM classifiers.
- **Automatic Model Selection:** Compares models and automatically serializes the best performing one to `.pkl` format.
- **Interactive Dashboard:** A Streamlit frontend for exploring the dataset, viewing model performance charts, and running predictions on new CSV data.
- **Zero-Retraining Inference:** The Streamlit app strictly loads the pre-trained model for fast inference without incurring training delays.

## Folder Structure
```text
credit crad frauad detection/
│
├── dataset/
│      ├── creditcard.csv       # (Needs to be downloaded)
│      └── sample_input.csv     # (Generated during training)
│
├── models/                     # (Generated during training)
│      ├── fraud_model.pkl
│      └── scaler.pkl
│
├── outputs/                    # (Generated during training)
│      ├── best_model_info.json
│      └── ... plots (.png)
│
├── app.py                      # Streamlit application
├── train_model.py              # ML training script
├── utils.py                    # Helper functions
├── config.py                   # Project configurations
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── LICENSE                     # MIT License
```

## Dataset Source
The project uses the Kaggle Credit Card Fraud Detection dataset.
Download it from: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

*Note: You must extract the `creditcard.csv` file and place it inside the `dataset/` directory before running the training script.*

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd credit crad frauad detection
   ```

2. **Create a Virtual Environment (Optional but recommended):**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## How to Train

Run the training pipeline to preprocess data, train models, and save the best model and scaler.

```bash
python train_model.py
```
*This process will evaluate all models and automatically save `fraud_model.pkl` and `scaler.pkl` to the `models/` directory, along with evaluation plots in the `outputs/` directory.*

## How to Run Streamlit

Once the model is trained and saved, you can launch the Streamlit dashboard:

```bash
streamlit run app.py
```
*The app will be available locally at `http://localhost:8501/`.*

## Machine Learning Workflow
1. **Data Preprocessing:** Handles missing values and duplicates.
2. **Train/Test Split:** Standard 80/20 split with stratification.
3. **Scaling:** Standardizes features using `StandardScaler`.
4. **Class Balancing:** Applies `SMOTE` strictly on the training set to combat massive class imbalance.
5. **Model Evaluation:** Models are scored on Accuracy, Precision, Recall, F1 Score, and ROC-AUC.
6. **Inference:** Web app loads serialized artifacts to classify user-uploaded CSV batches.

## Future Improvements
- Integrate Deep Learning models (e.g., Autoencoders, Neural Networks).
- Deploy the Streamlit app to Streamlit Cloud, AWS, or Heroku.
- Add an API endpoint using FastAPI for automated integrations.

## License
MIT License. See `LICENSE` for more information.

## Author
Senior Machine Learning & UI/UX Engineer
