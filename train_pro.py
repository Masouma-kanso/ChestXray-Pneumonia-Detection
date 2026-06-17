import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import joblib

from features import extract_features

TRAIN_CSV = "data/train.csv"
VAL_CSV   = "data/val.csv"
TEST_CSV  = "data/test.csv"

OUT_MODEL = "models/rf_model_pro.pkl"

def build_X(df):
    X_list = []
    bad = []
    for i, row in df.iterrows():
        try:
            X_list.append(extract_features(row["image_path"]))
        except Exception as e:
            bad.append((i, row["image_path"], str(e)))

    X = np.array(X_list)
    if bad:
        df_ok = df.drop(index=[i for i,_,_ in bad]).reset_index(drop=True)
    else:
        df_ok = df.reset_index(drop=True)

    return X, df_ok, bad

# 1) Load splits
train_df = pd.read_csv(TRAIN_CSV).dropna(subset=["image_path","label"]).copy()
val_df   = pd.read_csv(VAL_CSV).dropna(subset=["image_path","label"]).copy()
test_df  = pd.read_csv(TEST_CSV).dropna(subset=["image_path","label"]).copy()

# 2) Encode labels using ONLY train labels (استاندارد)
le = LabelEncoder()
le.fit(train_df["label"].astype(str))

# 3) Build features
X_train, train_ok, bad_train = build_X(train_df)
y_train = le.transform(train_ok["label"].astype(str))

X_val, val_ok, bad_val = build_X(val_df)
y_val = le.transform(val_ok["label"].astype(str))

X_test, test_ok, bad_test = build_X(test_df)
y_test = le.transform(test_ok["label"].astype(str))

print("✅ Features ready:")
print("Train:", X_train.shape, " Val:", X_val.shape, " Test:", X_test.shape)
print("Labels:", list(le.classes_))

# 4) Train model
model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# 5) Validate (برای چک سریع)
val_pred = model.predict(X_val)
print("\n=== Validation ===")
print("Confusion Matrix:\n", confusion_matrix(y_val, val_pred))
print(classification_report(y_val, val_pred, target_names=le.classes_))

# 6) Final test (فقط یک بار، گزارش نهایی)
test_pred = model.predict(X_test)
print("\n=== Test (Final) ===")
print("Confusion Matrix:\n", confusion_matrix(y_test, test_pred))
print(classification_report(y_test, test_pred, target_names=le.classes_))

# 7) Save
joblib.dump({"model": model, "label_encoder": le}, OUT_MODEL)
print(f"\n✅ Model saved to: {OUT_MODEL}")

# 8) Report bad images (اگر بود)
total_bad = len(bad_train) + len(bad_val) + len(bad_test)
if total_bad:
    print(f"\n⚠️ Could not read {total_bad} images. Samples:")
    for item in (bad_train + bad_val + bad_test)[:5]:
        print(" -", item)