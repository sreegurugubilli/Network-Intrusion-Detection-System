# Network Intrusion Detection System (Streamlit)

A dataset-based educational prototype that trains a machine-learning classifier on labeled network traffic CSV data and displays predictions and evaluation metrics.

## Features
- Upload a labeled CSV dataset
- Select the target/label column
- Train a Random Forest classifier
- View test accuracy, classification report, and confusion matrix
- Download predicted labels

## Requirements
- Python 3.10 or newer recommended

## Run locally
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt
streamlit run app.py
```

## Dataset format
Upload a CSV where each row represents a network-flow record. Include numeric/categorical feature columns and a target column such as `Label` containing at least two classes (for example, `BENIGN`, `DoS`, `PortScan`). Do not upload private or sensitive network data to a public repository.

## Limitations
This app analyzes uploaded records; it does not capture live network traffic. Evaluation results depend on the dataset and split, and should not be treated as proof of production-grade intrusion detection.
