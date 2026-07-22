import os

# Base Directory Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'dataset')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')

# File Paths
DATASET_PATH = os.path.join(DATA_DIR, 'creditcard.csv')
SAMPLE_INPUT_PATH = os.path.join(DATA_DIR, 'sample_input.csv')
MODEL_PATH = os.path.join(MODELS_DIR, 'fraud_model.pkl')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')

# Model Configurations
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Column names
TARGET_COL = 'Class'
TIME_COL = 'Time'
AMOUNT_COL = 'Amount'

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
