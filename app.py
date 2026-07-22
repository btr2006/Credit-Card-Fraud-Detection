import streamlit as st
import pandas as pd
import os
import json
import plotly.express as px
from config import DATASET_PATH, TARGET_COL, OUTPUTS_DIR, MODEL_PATH, SCALER_PATH
from utils import load_model, load_scaler

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Caching ---
@st.cache_data
def get_dataset(nrows=None):
    try:
        return pd.read_csv(DATASET_PATH, nrows=nrows)
    except Exception as e:
        return None

@st.cache_resource
def get_model():
    try:
        return load_model(MODEL_PATH)
    except Exception as e:
        return None

@st.cache_resource
def get_scaler():
    try:
        return load_scaler(SCALER_PATH)
    except Exception as e:
        return None

# --- Page Functions ---
def show_welcome():
    st.title("🛡️ Credit Card Fraud Detection System")
    st.markdown("""
    ### Welcome to the Credit Card Fraud Detection Dashboard!
    
    This application leverages advanced Machine Learning techniques to detect fraudulent credit card transactions. 
    It provides a comprehensive interface to analyze the dataset, view model metrics, and run predictions on new transaction data.
    
    #### Project Objectives:
    - 🔍 Analyze highly imbalanced transaction data.
    - 🧠 Utilize state-of-the-art ML models to classify transactions as **Genuine** or **Fraudulent**.
    - 📊 Provide interactive visualizations of the data and model performance.
    - 🚀 Offer a robust prediction engine without retraining latency.
    """)
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("📊 **Dataset Overview**\n\nExplore the raw data, summary statistics, and missing values.")
    with col2:
        st.warning("📈 **Visualizations**\n\nView charts for data distribution and model comparison.")
    with col3:
        st.success("📤 **Upload & Predict**\n\nUpload a CSV of transactions to predict fraud instantly using the pre-trained model.")

def show_dataset_overview():
    st.title("📊 Dataset Overview")
    df = get_dataset(nrows=50000)
    if df is None:
        st.error(f"Dataset not found at {DATASET_PATH}. Please ensure the dataset exists.")
    else:
        st.markdown("### Quick Statistics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", f"{df.shape[0]:,}")
        with col2:
            st.metric("Total Features", f"{df.shape[1] - 1}")
        with col3:
            fraud_count = df[TARGET_COL].sum()
            st.metric("Fraudulent Transactions", f"{fraud_count:,}")
        with col4:
            genuine_count = df.shape[0] - fraud_count
            st.metric("Genuine Transactions", f"{genuine_count:,}")
            
        st.markdown("---")
        st.markdown("### Data Quality Check")
        
        col1, col2 = st.columns(2)
        with col1:
            missing_vals = df.isnull().sum().sum()
            if missing_vals == 0:
                st.success(f"Missing Values: {missing_vals} ✅")
            else:
                st.error(f"Missing Values: {missing_vals} ❌")
                
        with col2:
            duplicates = df.duplicated().sum()
            if duplicates == 0:
                st.success(f"Duplicate Rows: {duplicates} ✅")
            else:
                st.warning(f"Duplicate Rows: {duplicates} ⚠️")
                
        st.markdown("---")
        st.markdown("### Dataset Preview (First 100 Rows)")
        st.dataframe(df.head(100))
        
        st.markdown("### Summary Statistics")
        st.dataframe(df.describe())

def show_visualizations():
    st.title("📈 Visualizations")
    st.markdown("### Pre-computed Model Plots")
    
    plots = [
        ("Model Comparison", "model_comparison.png"),
        ("Random Forest ROC Curve", "Random_Forest_roc.png"),
        ("Random Forest Confusion Matrix", "Random_Forest_cm.png"),
        ("Logistic Regression ROC Curve", "Logistic_Regression_roc.png"),
        ("Logistic Regression Confusion Matrix", "Logistic_Regression_cm.png")
    ]
    
    for title, filename in plots:
        filepath = os.path.join(OUTPUTS_DIR, filename)
        if os.path.exists(filepath):
            st.markdown(f"#### {title}")
            st.image(filepath, use_column_width=True)
            st.markdown("---")
            
    st.info("If plots are missing, please run the `train_model.py` script first.")

def show_model_information():
    st.title("🤖 Model Information")
    info_path = os.path.join(OUTPUTS_DIR, 'best_model_info.json')
    if not os.path.exists(info_path):
        st.warning("Model information not found. Please train the models first using `train_model.py`.")
    else:
        with open(info_path, 'r') as f:
            model_info = json.load(f)
            
        st.markdown(f"### Selected Best Model: **{model_info['model_name']}**")
        st.markdown(f"The best model was selected automatically based on the ROC-AUC score during the training phase.")
        
        st.markdown("#### Evaluation Metrics")
        metrics = model_info['metrics']
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Accuracy", f"{metrics.get('Accuracy', 0):.4f}")
        with col2:
            st.metric("Precision", f"{metrics.get('Precision', 0):.4f}")
        with col3:
            st.metric("Recall", f"{metrics.get('Recall', 0):.4f}")
        with col4:
            st.metric("F1 Score", f"{metrics.get('F1 Score', 0):.4f}")
        with col5:
            st.metric("ROC AUC", f"{metrics.get('ROC AUC', 0):.4f}")
            
        st.markdown("#### Training Details")
        st.write(f"- **Training Time:** {metrics.get('Training Time (s)', 'N/A')} seconds")
        st.write(f"- **Number of Samples Used:** {model_info.get('num_samples', 'N/A'):,}")
        st.write(f"- **SMOTE Applied:** {'Yes' if model_info.get('smote_applied') else 'No'}")
        
        with st.expander("View Features Used"):
            st.write(model_info.get('features_used', []))

def show_download_sample():
    st.title("📥 Download Sample CSV")
    st.markdown("""
    You can download sample CSV files containing different rows of data (without the Class column) 
    to test the prediction functionality.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    for idx, (col, n_rows) in enumerate(zip([col1, col2, col3], [20, 50, 100])):
        path = os.path.join(DATASET_PATH.replace('creditcard.csv', f'sample_input_{n_rows}.csv'))
        with col:
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    st.download_button(
                        label=f"Download {n_rows} Rows Sample",
                        data=f,
                        file_name=f"sample_input_{n_rows}.csv",
                        mime="text/csv",
                        type="primary" if idx == 0 else "secondary",
                        use_container_width=True
                    )
            else:
                st.warning(f"{n_rows} row sample missing.")
                
    st.markdown("---")
    st.markdown("#### Sample Data Previews")
    
    tab1, tab2, tab3 = st.tabs(["20 Rows", "50 Rows", "100 Rows"])
    
    with tab1:
        path_20 = os.path.join(DATASET_PATH.replace('creditcard.csv', 'sample_input_20.csv'))
        if os.path.exists(path_20):
            st.dataframe(pd.read_csv(path_20))
            
    with tab2:
        path_50 = os.path.join(DATASET_PATH.replace('creditcard.csv', 'sample_input_50.csv'))
        if os.path.exists(path_50):
            st.dataframe(pd.read_csv(path_50))
            
    with tab3:
        path_100 = os.path.join(DATASET_PATH.replace('creditcard.csv', 'sample_input_100.csv'))
        if os.path.exists(path_100):
            st.dataframe(pd.read_csv(path_100))

def show_upload_predict():
    st.title("📤 Upload CSV & 🔍 Predict Fraud")
    model = get_model()
    scaler = get_scaler()
    
    if model is None or scaler is None:
        st.error("Model or Scaler not found! Please ensure you have run `train_model.py` to generate `.pkl` files.")
    else:
        st.markdown("Upload a CSV file containing transaction data to predict whether they are fraudulent or genuine.")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
        
        if uploaded_file is not None:
            try:
                input_df = pd.read_csv(uploaded_file)
                st.success("File uploaded successfully!")
                
                # Validation
                info_path = os.path.join(OUTPUTS_DIR, 'best_model_info.json')
                if os.path.exists(info_path):
                    with open(info_path, 'r') as f:
                        model_info = json.load(f)
                    expected_columns = set(model_info['features_used'])
                    actual_columns = set(input_df.columns)
                    
                    missing_cols = expected_columns - actual_columns
                    if missing_cols:
                        st.error(f"Validation Error: Missing columns in uploaded CSV: {missing_cols}")
                    else:
                        valid = True
                else:
                    st.warning("Model info not found, skipping strict column validation.")
                    valid = True
                
                if 'valid' in locals() and valid:
                    st.markdown("#### Uploaded Data Preview")
                    st.dataframe(input_df.head())
                    
                    if st.button("Predict Fraud", type="primary"):
                        with st.spinner("Analyzing transactions..."):
                            # Preprocess
                            if TARGET_COL in input_df.columns:
                                X = input_df.drop(columns=[TARGET_COL])
                            else:
                                X = input_df
                                
                            # Scale
                            X_scaled = scaler.transform(X)
                            
                            # Predict
                            predictions = model.predict(X_scaled)
                            
                            if hasattr(model, "predict_proba"):
                                probabilities = model.predict_proba(X_scaled)[:, 1]
                            else:
                                decision_scores = model.decision_function(X_scaled)
                                probabilities = (decision_scores - decision_scores.min()) / (decision_scores.max() - decision_scores.min())
                                
                            # Prepare results
                            results_df = input_df.copy()
                            results_df['Prediction'] = ["Fraud" if p == 1 else "Genuine" for p in predictions]
                            results_df['Fraud Probability'] = [f"{prob:.4f}" for prob in probabilities]
                            
                            st.markdown("---")
                            st.title("📋 Prediction Results")
                            
                            fraud_detected = (predictions == 1).sum()
                            genuine_detected = len(predictions) - fraud_detected
                            
                            if fraud_detected > 0:
                                st.error(f"⚠️ Alert: {fraud_detected} fraudulent transactions detected!")
                            else:
                                st.success("✅ No fraudulent transactions detected in this batch.")
                                
                            # Display Metrics
                            st.markdown("### Prediction Summary")
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Total Transactions", f"{len(predictions):,}")
                            with col2:
                                st.metric("Genuine Transactions", f"{genuine_detected:,}")
                            with col3:
                                st.metric("Fraudulent Transactions", f"{fraud_detected:,}")
                                
                            # Display Chart using Plotly
                            st.markdown("### Distribution of Predictions")
                            
                            fig = px.pie(
                                results_df, 
                                names='Prediction', 
                                color='Prediction',
                                color_discrete_map={'Genuine': '#2ca02c', 'Fraud': '#d62728'},
                                title="Predicted Genuine vs Fraud Transactions",
                                hole=0.4
                            )
                            fig.update_traces(textposition='inside', textinfo='percent+label+value')
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                            st.markdown("### Detailed Records")
                            st.dataframe(results_df)
                            
                            # Download Results
                            csv_results = results_df.to_csv(index=False)
                            st.download_button(
                                label="Download Prediction Results CSV",
                                data=csv_results,
                                file_name="prediction_results.csv",
                                mime="text/csv",
                                type="secondary"
                            )
                            
            except Exception as e:
                st.error(f"Error processing the file: {str(e)}")

def show_about():
    st.title("ℹ️ About Project")
    st.markdown("""
    ### Credit Card Fraud Detection System
    
    This project is a production-quality, end-to-end Machine Learning pipeline and Streamlit dashboard built to identify fraudulent credit card transactions.
    
    #### Key Features:
    - **Modular Architecture:** Clean separation between training, configuration, utilities, and application code.
    - **Advanced Imbalance Handling:** Uses SMOTE (Synthetic Minority Over-sampling Technique) to address the highly skewed dataset.
    - **Model Comparison:** Automatically evaluates Logistic Regression, Decision Trees, Random Forests, and SVMs.
    - **Inference App:** A lightweight Streamlit frontend that serves predictions instantaneously by loading serialized `.pkl` models.
    - **Professional UI:** Includes success/error states, metrics cards, wide layouts, and responsive dataframes.
    
    #### Technology Stack:
    - **Python 3**
    - **Pandas & NumPy** (Data Manipulation)
    - **Scikit-Learn & Imbalanced-Learn** (Machine Learning & SMOTE)
    - **Matplotlib, Seaborn & Plotly** (Data Visualization)
    - **Streamlit** (Web Dashboard)
    
    #### Author:
    Developed as an industry-standard portfolio project demonstrating best practices in Machine Learning Engineering and Software Development.
    """)

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
menu_options = [
    "🏠 Welcome",
    "📊 Dataset Overview",
    "📈 Visualizations",
    "🤖 Model Information",
    "📥 Download Sample",
    "📤 Upload & Predict",
    "ℹ️ About Project"
]
selection = st.sidebar.radio("Go to", menu_options)

# --- Routing ---
if selection == "🏠 Welcome":
    show_welcome()
elif selection == "📊 Dataset Overview":
    show_dataset_overview()
elif selection == "📈 Visualizations":
    show_visualizations()
elif selection == "🤖 Model Information":
    show_model_information()
elif selection == "📥 Download Sample":
    show_download_sample()
elif selection == "📤 Upload & Predict":
    show_upload_predict()
elif selection == "ℹ️ About Project":
    show_about()
