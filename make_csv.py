import os
import csv

base_path = r"C:\Users\pc\Downloads\chest_xray\train"

data = []

for label in ["NORMAL", "PNEUMONIA"]:
    folder = os.path.join(base_path, label)

    for img_name in os.listdir(folder):
        if img_name.endswith(".jpeg") or img_name.endswith(".jpg"):
            path = os.path.join(folder, img_name)
            data.append([path, label])

with open("data/dataset.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "label"])
    writer.writerows(data)

print("CSV created successfully!")
