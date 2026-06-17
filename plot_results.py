import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from features import extract_features

VAL_CSV  = "data/val.csv"
TEST_CSV = "data/test.csv"
MODEL_PATH = "models/rf_model_pro.pkl"

os.makedirs("outputs", exist_ok=True)

# Load model + label encoder
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
le = bundle["label_encoder"]

def build_X(df):
    X_list = []
    ok_labels = []
    bad = []
    for i, row in df.iterrows():
        try:
            X_list.append(extract_features(row["image_path"]))
            ok_labels.append(row["label"])
        except Exception as e:
            bad.append((i, row["image_path"], str(e)))
    X = np.array(X_list)
    y = le.transform(pd.Series(ok_labels).astype(str))
    return X, y, bad

def evaluate_split(name, csv_path):
    df = pd.read_csv(csv_path).dropna(subset=["image_path","label"]).copy()
    X, y, bad = build_X(df)

    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)

    # Plot confusion matrix
    plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(values_format="d")
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(f"outputs/confusion_{name.lower()}.png")
    plt.close()

    # Save report
    report_txt = classification_report(y, y_pred, target_names=le.classes_)
    with open(f"outputs/report_{name.lower()}.txt", "w", encoding="utf-8") as f:
        f.write(report_txt)

    print(f"✅ Saved outputs/confusion_{name.lower()}.png and outputs/report_{name.lower()}.txt")
    if bad:
        print(f"⚠️ {name}: {len(bad)} images could not be read. Sample:", bad[:2])

evaluate_split("VAL", VAL_CSV)
evaluate_split("TEST", TEST_CSV) 