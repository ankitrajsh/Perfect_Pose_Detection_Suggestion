
# 📸AI-Powered Pose Detection and Recomendation & Outfit Assistant

Pose Detection and Recomendation is an AI-based assistant that helps users take the perfect photo by recommending context-aware poses, camera settings, and outfits based on their environment, lighting, and event type. It acts as a personal stylist and smart photographer combined into one intelligent application.

---

## 🚀 Features

- 🔍 **Scene Detection** – Classifies the environment (beach, park, indoor, etc.) using deep learning.
- 💡 **Smart Lighting & Camera Suggestions** – Recommends custom camera settings like ISO, exposure, and white balance.
- 🧍 **Pose Detection & Recommendation** – Suggests best poses based on detected scene and lighting using MediaPipe.
- 👗 **Outfit Recommendation** – Recommends appropriate outfits based on theme, weather, and event using a fashion-aware model.
- 🧠 **Personalized Suggestions** – Learns your preferences over time and adapts recommendations accordingly.
- 🎨 **Mock Preview Generator** – Shows you how you’d look in the suggested pose and outfit using generative AI.
- 📷 **Real-time Overlay** – Guides you visually during the photo shoot with pose outlines and visual indicators.

---

## 🧠 AI Model Architecture

The project is divided into 8 modular components:

| Module                         | Description |
|-------------------------------|-------------|
| **1. User Context Detection** | Gathers location, time, event info, and environmental lighting. |
| **2. Scene Recognition** | Classifies environment using a pre-trained ViT model or Places365. |
| **3. Camera Estimation** | Suggests camera settings based on ambient light and scene. |
| **4. Pose Estimation & Suggestion** | Detects user pose and recommends improvements using MediaPipe. |
| **5. Outfit Recommendation** | Suggests clothes based on fashion compatibility using CLIP-based vector similarity. |
| **6. Personalization Engine** | Adapts suggestions based on user history and preferences. |
| **7. AI Preview Generator** | Creates image previews using generative models (e.g., Stable Diffusion). |
| **8. Output Interface** | Displays visual guidance, final suggestions, and captures results. |

---

## 📂 Project Structure

```
PosePal/
├── outfits/                   # Outfit images for matching
├── modules/                  # Individual AI modules (scene_detect, pose_detect, etc.)
├── utils/                    # Helper functions and utilities
├── app.py                    # Main script or Streamlit/Flask app
├── requirements.txt          # Dependencies
├── README.md                 # Project documentation
└── outfit_db.json            # Outfit metadata with tags
```

---

## 🧰 Requirements

Install the necessary dependencies:

```bash
pip install -r requirements.txt
```

> Ensure you have a CUDA-compatible GPU if using Stable Diffusion or other heavy models.

---

## 📸 Sample Outfit Database

Each outfit has an image and tag metadata:

```python
OUTFIT_DB = [
    {"image_path": "outfits/casual_blue_jeans.jpg", "tags": ["casual", "blue", "jeans", "outdoor", "day"]},
    {"image_path": "outfits/formal_black_suit.jpg", "tags": ["formal", "black", "suit", "indoor", "evening"]},
    ...
]
```

Download free outfit images from:
- [Pexels](https://www.pexels.com)
- [Pixabay](https://www.pixabay.com)
- [Unsplash](https://www.unsplash.com)

---

## 🖼️ Demo Preview

> Coming soon! 🎥 A demo video showing pose and outfit suggestions in real-time.

---

## 🤖 Future Enhancements

- Voice control and smart mirror integration
- Real-time virtual try-on using AR
- Deeper personalization via user profiling
- Event-driven outfit planner (calendar + weather forecast)

---

## 🙌 Contributing

We welcome contributions! Submit a PR or open an issue for suggestions.

---

## 📄 License

MIT License – see `LICENSE` file for details.

---

## ✨ Credits

- [MediaPipe](https://mediapipe.dev)
- [CLIP by OpenAI](https://github.com/openai/CLIP)
- [HuggingFace Transformers](https://huggingface.co)
- [Pexels](https://pexels.com) / [Unsplash](https://unsplash.com) for outfit imagery
