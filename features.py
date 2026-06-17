import cv2
import numpy as np
from skimage.feature import hog

IMG_SIZE = (64, 64)

def extract_features_from_array(img_gray: np.ndarray):
   
    img = cv2.resize(img_gray, IMG_SIZE)
    features = hog(
        img,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=False
    )
    return features

def extract_features(image_path: str):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    return extract_features_from_array(img)