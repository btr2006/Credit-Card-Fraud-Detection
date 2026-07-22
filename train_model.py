import os
import pandas as pd
import numpy as np
import time
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from imblearn.over_sampling import SMOTE

from config import DATASET_PATH, SAMPLE_INPUT_PATH, MODEL_PATH, SCALER_PATH, OUTPUTS_DIR, TARGET_COL, TEST_SIZE, RANDOM_STATE
from utils import load_data, save_model, save_scaler, evaluate_model, plot_confusion_matrix, plot_roc_curve, plot_precision_recall_curve, plot_model_comparison

def prepare_data(df: pd.DataFrame):
    """Preprocesses the dataset: handles missing values, duplicates, and splits features/target."""
    print("--- Data Preprocessing ---")
    
    # 1. Missing value check
    missing_count = df.isnull().sum().sum()
    if missing_count > 0:
        print(f"Found {missing_count} missing values. Dropping rows with missing values.")
        df = df.dropna()
    else:
        print("No missing values found.")

    # 2. Duplicate check
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        print(f"Found {duplicate_count} duplicate rows. Dropping duplicates.")
        df = df.drop_duplicates()
    else:
        print("No duplicate rows found.")

    # Separate features and target
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    return X, y

def create_sample_csv(df: pd.DataFrame, base_path: str):
    """Generates a sample input CSV from the dataset."""
    print(f"\n--- Generating Sample CSVs ---")
    for n in [20, 50, 100]:
        path = base_path.replace('sample_input.csv', f'sample_input_{n}.csv')
        sample_df = df.drop(columns=[TARGET_COL]).head(n)
        sample_df.to_csv(path, index=False)
        print(f"Sample CSV ({n} rows) saved to {path}")

def train_and_evaluate():
    """Main function to train models, evaluate, and save the best one."""
    
    # Ensure outputs dir exists
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset from {DATASET_PATH}...")
    try:
        df = load_data(DATASET_PATH)
    except FileNotFoundError as e:
        print(e)
        return

    print(f"Dataset shape: {df.shape}")
    
    # Generate sample CSV
    create_sample_csv(df, SAMPLE_INPUT_PATH)

    # Preprocess
    X, y = prepare_data(df)

    # Train-test split
    print("\n--- Splitting Data ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # Scaling
    print("\n--- Feature Scaling ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # SMOTE (only on training data)
    print("\n--- Applying SMOTE ---")
    print(f"Class distribution before SMOTE: {y_train.value_counts().to_dict()}")
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    print(f"Class distribution after SMOTE: {y_train_resampled.value_counts().to_dict()}")

    # Define models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'Decision Tree': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE, n_jobs=-1),
        'SVM': LinearSVC(dual=False, random_state=RANDOM_STATE, max_iter=2000)
    }

    results = {}
    best_model_name = None
    best_roc_auc = -1
    best_model = None

    print("\n--- Training Models ---")
    for name, model in models.items():
        print(f"Training {name}...")
        start_time = time.time()
        
        # Train
        model.fit(X_train_resampled, y_train_resampled)
        train_time = time.time() - start_time
        
        # Predict
        y_pred = model.predict(X_test_scaled)
        
        # Probabilities for ROC AUC (SVM requires probability=True)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
        else:
            y_prob = model.decision_function(X_test_scaled)
            
        # Evaluate
        metrics = evaluate_model(y_test, y_pred, y_prob)
        metrics['Training Time (s)'] = round(train_time, 2)
        results[name] = metrics
        
        print(f"{name} metrics: {metrics}")
        
        # Generate plots
        plot_confusion_matrix(y_test, y_pred, title=f"{name} Confusion Matrix", save_path=os.path.join(OUTPUTS_DIR, f"{name.replace(' ', '_')}_cm.png"))
        plot_roc_curve(y_test, y_prob, title=f"{name} ROC Curve", save_path=os.path.join(OUTPUTS_DIR, f"{name.replace(' ', '_')}_roc.png"))
        plot_precision_recall_curve(y_test, y_prob, title=f"{name} PR Curve", save_path=os.path.join(OUTPUTS_DIR, f"{name.replace(' ', '_')}_pr.png"))

        # Select best model based on ROC AUC
        if metrics['ROC AUC'] > best_roc_auc:
            best_roc_auc = metrics['ROC AUC']
            best_model_name = name
            best_model = model

    print("\n--- Model Comparison ---")
    results_df = pd.DataFrame(results).T
    print(results_df)
    
    # Save model comparison plot
    plot_model_comparison(results_df.drop(columns=['Training Time (s)']), save_path=os.path.join(OUTPUTS_DIR, "model_comparison.png"))
    
    # Save best model information
    print(f"\nBest Model Selected: {best_model_name} with ROC AUC = {best_roc_auc:.4f}")
    
    model_info = {
        'model_name': best_model_name,
        'metrics': results[best_model_name],
        'features_used': list(X.columns),
        'num_samples': len(df),
        'smote_applied': True
    }
    
    with open(os.path.join(OUTPUTS_DIR, 'best_model_info.json'), 'w') as f:
        json.dump(model_info, f, indent=4)

    print(f"\n--- Saving Best Model & Scaler ---")
    save_model(best_model, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")
    
    save_scaler(scaler, SCALER_PATH)
    print(f"Scaler saved to {SCALER_PATH}")
    
    print("\nTraining completed successfully!")

if __name__ == "__main__":
    train_and_evaluate()
