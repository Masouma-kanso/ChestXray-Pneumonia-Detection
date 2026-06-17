# Chest X-ray Pneumonia Detection

A Machine Learning project for detecting pneumonia from chest X-ray images.

## Features
- Image feature extraction using OpenCV
- Random Forest classification
- Train / Validation / Test split
- Performance evaluation & confusion matrix
- Heatmap visualization (occlusion sensitivity)
- Streamlit web app demo

## Project Structure
ChestXrayProject/
│
├── data/
├── models/
├── outputs/
├── features.py
├── train_pro.py
└── app.py

## Installation

Install required libraries:

pip install numpy pandas scikit-learn opencv-python streamlit joblib matplotlib

## Train Model

Run:

python train_pro.py

## Run Web App

streamlit run app.py

## Usage

1. Open the web app in your browser  
2. Upload a chest X-ray image  
3. View prediction and heatmap visualization    

