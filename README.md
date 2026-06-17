# Chest X-ray Pneumonia Detection

A machine learning project for detecting **pneumonia** from chest X-ray images using classical image processing, HOG feature extraction, and a Random Forest classifier.

> This project is developed for educational and research purposes only. It is not intended for real medical diagnosis or clinical decision-making.

---

## Overview

Pneumonia is a serious lung infection that can be detected from chest X-ray images.  
This project builds a complete machine learning pipeline to classify chest X-ray images into two classes:

- **NORMAL**
- **PNEUMONIA**

The project includes dataset preparation, exploratory data analysis, feature extraction, model training, evaluation, and a Streamlit web application for demonstration.

---

## Features

- Chest X-ray image classification
- Image preprocessing using OpenCV
- HOG feature extraction
- Random Forest classifier
- Train / Validation / Test data split
- Model evaluation using classification report
- Confusion matrix visualization
- Streamlit web application
- Prediction confidence display
- Occlusion sensitivity heatmap for visual explanation

---

## Project Structure

```text
ChestXrayProject/
│
├── data/
│   ├── dataset.csv
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
│
├── models/
│   └── rf_model_pro.pkl
│
├── outputs/
│   ├── confusion_val.png
│   ├── confusion_test.png
│   ├── report_val.txt
│   └── report_test.txt
│
├── app.py
├── features.py
├── make_csv.py
├── eda_split.py
├── train.py
├── train_pro.py
├── plot_results.py
├── ChestXray_EDA_and_Split.ipynb
└── README.md
