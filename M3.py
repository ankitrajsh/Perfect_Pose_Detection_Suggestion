import cv2
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import joblib
import os

# -------- Feature Extraction from Image --------
def extract_image_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or path is incorrect.")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    contrast = gray.std()

    r, g, b = cv2.split(image)
    avg_r, avg_g, avg_b = np.mean(r), np.mean(g), np.mean(b)
    color_temp = (avg_r - avg_b)

    features = np.array([brightness, contrast, avg_r, avg_g, avg_b, color_temp])
    return features.reshape(1, -1)

# -------- Dummy Training (Load Pretrained Model in Production) --------
def train_or_load_model(model_path='camera_model.pkl', scaler_path='scaler.pkl'):
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler

    # Dummy dataset (synthetic - for illustration)
    X = np.random.rand(200, 6) * [255, 100, 255, 255, 255, 100]  # brightness, contrast, RGB, color temp
    y_iso = np.clip((X[:, 0] + X[:, 1]) * 2, 100, 800)
    y_wb = 5000 + (X[:, 5] * 10)
    y_exp = 1/np.clip((X[:, 0] / 25 + 1), 1, 200)
    y_ap = np.clip((X[:, 1] / 20 + 2), 1.4, 8.0)

    y = np.stack([y_iso, y_wb, y_exp, y_ap], axis=1)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = lgb.LGBMRegressor()
    model.fit(X_scaled, y)

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    return model, scaler

# -------- Suggest Camera Settings --------
def suggest_camera_settings(image_path):
    features = extract_image_features(image_path)
    model, scaler = train_or_load_model()
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    iso, white_balance, exposure, aperture = prediction

    result = {
        "ISO": int(round(iso, -1)),
        "White Balance (K)": int(round(white_balance)),
        "Shutter Speed (s)": round(exposure, 4),
        "Aperture (f/)": round(aperture, 1)
    }
    return result

# --------- Main Function ---------
if __name__ == "__main__":
    test_image = "temp.jpg"  # Replace with your image path
    try:
        suggestions = suggest_camera_settings(test_image)
        print("📷 Suggested Camera Settings:")
        for k, v in suggestions.items():
            print(f"  {k}: {v}")
    except Exception as e:
        print(f"Error: {e}")
