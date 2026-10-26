import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Network Intrusion Detection", page_icon="🛡️", layout="wide")

st.title("🛡️ Network Intrusion Detection System")
st.caption("Automatically loads the sample network-traffic dataset from the dataset folder.")

with st.sidebar:
    st.header("Instructions")
    st.markdown(
        "1. The app loads a CSV from the dataset folder automatically.\n"
        "2. Choose the column containing the traffic label.\n"
        "3. Train and evaluate the model.\n"
        "4. Review predictions and download results."
    )
    st.warning("This is a dataset-based demo, not a live network monitor.")

from pathlib import Path

# Load a CSV from the project's dataset folder by default.
DATASET_DIR = Path(__file__).parent / "dataset"
local_csv_files = sorted(DATASET_DIR.glob("*.csv")) if DATASET_DIR.exists() else []

try:
    if local_csv_files:
        default_csv = local_csv_files[0]
        df = pd.read_csv(default_csv, low_memory=False)
        st.success(f"Dataset loaded automatically: {default_csv.name}")
        if len(local_csv_files) > 1:
            st.info(f"Found {len(local_csv_files)} CSV files. Using {default_csv.name}.")
    else:
        st.error("No CSV file found. Put sample_network_intrusion_dataset.csv inside the dataset folder next to app.py.")
        st.stop()
except Exception as exc:
    st.error(f"Could not read the dataset: {exc}")
    st.stop()

if df.empty:
    st.error("The uploaded CSV is empty.")
    st.stop()

st.subheader("Dataset preview")
st.write(f"Rows: **{len(df):,}** · Columns: **{len(df.columns)}**")
st.dataframe(df.head(20), use_container_width=True)

label_candidates = [c for c in df.columns if c.strip().lower() in ("label", "class", "target", "attack", "result")]
default_idx = df.columns.get_loc(label_candidates[0]) if label_candidates else len(df.columns) - 1
label_col = st.selectbox("Select the label/target column", df.columns, index=default_idx)

if df[label_col].nunique(dropna=True) < 2:
    st.error("The selected label column needs at least two distinct classes.")
    st.stop()

# Keep a copy for predictions, then prepare training data.
work = df.dropna(subset=[label_col]).copy()
y_raw = work[label_col].astype(str).str.strip()
X = work.drop(columns=[label_col]).copy()

# Remove columns that are empty or unique per row (often identifiers, which can cause leakage).
drop_cols = []
for col in X.columns:
    if X[col].isna().all() or X[col].nunique(dropna=True) <= 1:
        drop_cols.append(col)
X = X.drop(columns=drop_cols, errors="ignore")

if X.shape[1] == 0:
    st.error("No usable feature columns remain after removing empty/constant columns.")
    st.stop()

# Convert numeric-looking object columns when possible.
for col in X.columns:
    if X[col].dtype == "object":
        converted = pd.to_numeric(X[col], errors="coerce")
        if converted.notna().mean() >= 0.90:
            X[col] = converted

# Remove likely row identifiers to reduce accidental memorization.
id_like = [c for c in X.columns if c.strip().lower() in ("id", "flow_id", "timestamp", "src_ip", "source_ip", "dst_ip", "destination_ip")]
X = X.drop(columns=id_like, errors="ignore")

if X.shape[1] == 0:
    st.error("No usable feature columns remain after removing likely identifiers.")
    st.stop()

class_counts = y_raw.value_counts()
st.subheader("Label distribution")
st.bar_chart(class_counts)

if class_counts.min() < 2:
    st.error("Each class needs at least 2 rows to create a stratified train/test split.")
    st.stop()

test_size = st.slider("Test data percentage", min_value=10, max_value=40, value=20, step=5) / 100
train_button = st.button("🚀 Train and evaluate model", type="primary")

if train_button:
    with st.spinner("Training model…"):
        try:
            # Encode labels as integers for consistent reporting.
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y_raw)

            numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
            categorical_cols = [c for c in X.columns if c not in numeric_cols]

            numeric_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])
            categorical_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ])
            transformers = []
            if numeric_cols:
                transformers.append(("numeric", numeric_pipe, numeric_cols))
            if categorical_cols:
                transformers.append(("categorical", categorical_pipe, categorical_cols))
            preprocess = ColumnTransformer(transformers=transformers)

            model = Pipeline([
                ("preprocess", preprocess),
                ("classifier", RandomForestClassifier(
                    n_estimators=100,
                    random_state=42,
                    class_weight="balanced",
                    n_jobs=-1
                ))
            ])

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            acc = accuracy_score(y_test, pred)

            st.session_state["nids_model"] = model
            st.session_state["nids_encoder"] = label_encoder
            st.session_state["nids_features"] = X.columns.tolist()
            st.session_state["nids_label_col"] = label_col
            st.session_state["nids_accuracy"] = acc
            st.session_state["nids_report"] = classification_report(
                y_test, pred, labels=np.arange(len(label_encoder.classes_)),
                target_names=label_encoder.classes_, zero_division=0, output_dict=True
            )
            st.session_state["nids_cm"] = confusion_matrix(
                y_test, pred, labels=np.arange(len(label_encoder.classes_))
            )
            st.session_state["nids_classes"] = label_encoder.classes_.tolist()
            st.session_state["nids_test_y"] = y_test
            st.session_state["nids_test_pred"] = pred
            st.session_state["nids_test_X"] = X_test.copy()
            st.success("Model training and evaluation completed.")
        except Exception as exc:
            st.error(f"Training failed: {exc}")
            st.stop()

if "nids_model" in st.session_state:
    st.subheader("Model evaluation")
    c1, c2, c3 = st.columns(3)
    c1.metric("Test accuracy", f"{st.session_state['nids_accuracy']:.1%}")
    c2.metric("Training rows", f"{len(work) - len(st.session_state['nids_test_y']):,}")
    c3.metric("Test rows", f"{len(st.session_state['nids_test_y']):,}")

    report_df = pd.DataFrame(st.session_state["nids_report"]).T
    st.markdown("**Classification report**")
    st.dataframe(report_df.round(3), use_container_width=True)

    st.markdown("**Confusion matrix**")
    cm = st.session_state["nids_cm"]
    classes = st.session_state["nids_classes"]
    cm_df = pd.DataFrame(cm, index=[f"Actual: {c}" for c in classes],
                         columns=[f"Predicted: {c}" for c in classes])
    st.dataframe(cm_df, use_container_width=True)

    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Predict labels for uploaded records")
    st.caption("Predictions below are for the uploaded rows. If the label column is present, it is excluded from model input.")
    predict_df = df.drop(columns=[label_col], errors="ignore").copy()
    features = st.session_state["nids_features"]

    # Ensure same feature columns/order as training.
    for col in features:
        if col not in predict_df.columns:
            predict_df[col] = np.nan
    predict_df = predict_df[features]

    try:
        predictions = st.session_state["nids_model"].predict(predict_df)
        decoded = st.session_state["nids_encoder"].inverse_transform(predictions)
        results = df.copy()
        results["Predicted_Label"] = decoded
        benign_terms = {"benign", "normal", "0", "benign traffic"}
        results["Detection"] = [
            "Normal" if str(label).strip().lower() in benign_terms else "Potentially malicious"
            for label in decoded
        ]
        normal_count = int((results["Detection"] == "Normal").sum())
        suspicious_count = len(results) - normal_count
        d1, d2 = st.columns(2)
        d1.metric("Predicted normal", f"{normal_count:,}")
        d2.metric("Potentially malicious", f"{suspicious_count:,}")
        st.dataframe(results.head(1000), use_container_width=True)
        st.download_button(
            "⬇️ Download prediction results",
            data=results.to_csv(index=False).encode("utf-8"),
            file_name="nids_predictions.csv",
            mime="text/csv"
        )
    except Exception as exc:
        st.error(f"Could not generate predictions: {exc}")

st.caption("Educational prototype. Results depend on dataset quality and may not generalize to real networks.")
