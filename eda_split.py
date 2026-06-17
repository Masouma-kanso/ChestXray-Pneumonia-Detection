import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

CSV_PATH = "data/dataset.csv"   

df = pd.read_csv(CSV_PATH).dropna(subset=["image_path", "label"]).copy()

counts = df["label"].value_counts()
print("Class counts:\n", counts)

plt.figure()
counts.plot(kind="bar")
plt.title("Class Distribution (Bar)")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

plt.figure()
counts.plot(kind="pie", autopct="%1.1f%%")
plt.title("Class Distribution (Pie)")
plt.ylabel("")  
plt.show()

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"]
)

train_df, val_df = train_test_split(
    train_df,
    test_size=0.2,          
    random_state=42,
    stratify=train_df["label"]
)

train_df.to_csv("data/train.csv", index=False)
val_df.to_csv("data/val.csv", index=False)
test_df.to_csv("data/test.csv", index=False)

print("\n✅ Saved:")
print(" - data/train.csv:", len(train_df))
print(" - data/val.csv  :", len(val_df))
print(" - data/test.csv :", len(test_df))

print("\nTrain distribution:\n", train_df["label"].value_counts(normalize=True))
print("\nVal distribution:\n", val_df["label"].value_counts(normalize=True))
print("\nTest distribution:\n", test_df["label"].value_counts(normalize=True))