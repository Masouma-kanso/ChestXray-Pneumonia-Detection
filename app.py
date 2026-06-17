import streamlit as st
import numpy as np
import joblib
import cv2

from features import extract_features, extract_features_from_array

MODEL_PATH = "models/rf_model_pro.pkl"

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
le = bundle["label_encoder"]


def occlusion_heatmap(img_gray, model, target_class_idx, grid=8):
    base_img = cv2.resize(img_gray, (64, 64))

    base_feat = extract_features_from_array(base_img)
    base_proba = model.predict_proba([base_feat])[0][target_class_idx]

    h, w = base_img.shape
    heat = np.zeros((h, w), dtype=np.float32)

    ph = h // grid
    pw = w // grid
    mean_val = float(base_img.mean())

    for r in range(grid):
        for c in range(grid):
            y1, y2 = r * ph, (r + 1) * ph if r < grid - 1 else h
            x1, x2 = c * pw, (c + 1) * pw if c < grid - 1 else w

            occluded = base_img.copy()
            occluded[y1:y2, x1:x2] = mean_val

            feat = extract_features_from_array(occluded)
            proba = model.predict_proba([feat])[0][target_class_idx]

            drop = base_proba - proba
            heat[y1:y2, x1:x2] = drop

    heat = heat - heat.min()
    if heat.max() > 0:
        heat = heat / heat.max()

    return heat


st.title("Chest X-ray Pneumonia Detection (Demo)")

uploaded = st.file_uploader("Upload a Chest X-ray image", type=["jpg", "jpeg", "png"])

if uploaded is not None:

    temp_path = "temp_upload.png"
    with open(temp_path, "wb") as f:
        f.write(uploaded.getbuffer())

    st.image(temp_path, caption="Uploaded X-ray", use_container_width=True)

    feat = extract_features(temp_path)
    X = np.array([feat])

    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0] if hasattr(model, "predict_proba") else None
    label = le.inverse_transform([pred])[0]

    st.subheader("Prediction")
    st.write(f"**{label}**")

    if proba is not None:
        probs = {le.classes_[i]: float(proba[i]) for i in range(len(le.classes_))}
        st.subheader("Confidence")
        st.json(probs)

    show_hm = st.checkbox("Show Heatmap (Occlusion)")
    if show_hm:
        img_gray = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)

        if "PNEUMONIA" in list(le.classes_):
            target_idx = list(le.classes_).index("PNEUMONIA")
        else:
            target_idx = int(pred)

        heat = occlusion_heatmap(img_gray, model, target_idx, grid=8)

        h0, w0 = img_gray.shape
        heat_big = cv2.resize(heat, (w0, h0))
        heat_color = cv2.applyColorMap((heat_big * 255).astype("uint8"), cv2.COLORMAP_JET)

        img_bgr = cv2.imread(temp_path)
        overlay = cv2.addWeighted(img_bgr, 0.6, heat_color, 0.4, 0)

        st.subheader("Heatmap Overlay")
        st.image(overlay, channels="BGR", use_container_width=True)
        st.caption("Red/Yellow =(Occlusion Sensitivity)")