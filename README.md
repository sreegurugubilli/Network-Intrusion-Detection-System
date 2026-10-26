# 🛡️ Network Intrusion Detection System

A machine-learning-based Network Intrusion Detection System built using Python and Streamlit. The application loads a network traffic dataset, trains a Random Forest classifier, evaluates model performance, and predicts whether traffic is normal or potentially malicious.

**Note:** This is an educational prototype that uses synthetic demonstration data. It does not monitor live network traffic.

## Features

- Automatically loads a CSV dataset from the `dataset` folder.
- Displays a dataset preview.
- Shows the distribution of traffic labels.
- Allows selection of the target label column.
- Splits data into training and testing sets.
- Trains a Random Forest classification model.
- Displays test accuracy and a classification report.
- Shows the confusion matrix.
- Predicts labels for network traffic records.
- Displays normal and potentially malicious traffic counts.
- Allows users to download prediction results as a CSV file.

## Technologies Used

- Python
- Streamlit
- pandas
- NumPy
- scikit-learn
- Matplotlib

## Project Structure

```text
Network-Intrusion-Detection-System/
├── app.py
├── requirements.txt
├── README.md
└── dataset/
    └── sample_network_intrusion_dataset.csv
```

## Installation and Execution

### 1. Clone the repository

```bash
git clone https://github.com/sreegurugubill/Network-Intrusion-Detection-System.git
```

### 2. Open the project folder

```bash
cd Network-Intrusion-Detection-System
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, usually `http://localhost:8501`.

## How It Works

1. The application loads a CSV file from the `dataset` folder.
2. The user selects the target label column.
3. The application displays the dataset preview and label distribution.
4. The data is divided into training and testing sets.
5. A Random Forest classifier is trained on the training data.
6. The application evaluates the model using test accuracy, a classification report, and a confusion matrix.
7. The trained model predicts labels for the dataset records.
8. The prediction results can be downloaded as a CSV file.

## Dataset

The sample dataset contains synthetic network traffic records created for demonstration purposes. It is not a verified collection of real network traffic or attacks.

To use another dataset, place a compatible CSV file in the `dataset` folder. The CSV should contain feature columns and a target column with at least two distinct classes.

## Limitations

- The application does not capture or monitor live network traffic.
- Model performance depends on the dataset and its quality.
- Results from synthetic data do not establish real-world detection accuracy.
- This project is intended for educational and demonstration purposes.

## Conclusion

This project demonstrates how machine learning can be applied to network traffic classification through a simple Streamlit dashboard. It provides dataset visualization, model evaluation, traffic label prediction, and downloadable results.
