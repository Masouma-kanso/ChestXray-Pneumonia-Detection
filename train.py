import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import joblib

from features import extract_features

CSV_PATH = "data/dataset.csv"          
MODEL_PATH = "models/rf_model.pkl"    

df = pd.read_csv(CSV_PATH)

df = df.dropna(subset=["image_path", "label"]).copy()

X_list = []
bad_rows = []

for i, row in df.iterrows():
    try:
        X_list.append(extract_features(row["image_path"]))
    except Exception as e:
        bad_rows.append((i, row["image_path"], str(e)))

X = np.array(X_list)

if bad_rows:
    df_ok = df.drop(index=[i for i,_,_ in bad_rows]).reset_index(drop=True)
else:
    df_ok = df.reset_index(drop=True)

le = LabelEncoder()
y = le.fit_transform(df_ok["label"].astype(str))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Labels:", list(le.classes_))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))

joblib.dump({"model": model, "label_encoder": le}, MODEL_PATH)
print(f"\n✅ Model saved to: {MODEL_PATH}")

if bad_rows:
    print(f"\n⚠️ {len(bad_rows)} images could not be read. Sample:")
    for r in bad_rows[:5]:
        print(" -", r)