import os
import torch
import clip
from PIL import Image
from typing import List, Dict

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load CLIP model
model, preprocess = clip.load("ViT-B/32", device=device)

# Example outfit database (folder of images + metadata)
OUTFIT_DB = [
    {"image_path": "C:/Users/Nishith/Download/casual_blue_jeans.jpg", "tags": ["casual", "blue", "jeans", "outdoor", "day"]},
    {"image_path": "C:/Users/Nishith/Downloads/formal_black_suit.jpg", "tags": ["formal", "black", "suit", "indoor", "evening"]},
    {"image_path": "C:/Users/Nishith/Downloads/winter_coat.jpg", "tags": ["casual", "white", "dress", "outdoor", "summer", "day"]},
    {"image_path": "C:/Users/Nishith/Downloads/summer_white_dress.jpg", "tags": ["casual", "coat", "winter", "outdoor", "cold"]},
    # Add more as needed
]

# Rule-based color palettes based on scene and occasion
COLOR_PALETTES = {
    "formal": ["black", "navy", "white", "grey"],
    "casual": ["blue", "white", "green", "beige", "brown"],
    "party": ["red", "gold", "black", "silver"],
    "summer": ["white", "yellow", "light blue", "pastel pink"],
    "winter": ["dark green", "maroon", "navy", "brown"],
}

# Basic outfit style suggestions based on inputs
def suggest_styles(scene_type: str, occasion: str, weather: str = None) -> List[str]:
    styles = []
    if occasion in ["formal", "business"]:
        styles.append("formal")
    elif occasion in ["party", "wedding"]:
        styles.append("party")
    else:
        styles.append("casual")

    if weather in ["cold", "winter"]:
        styles.append("winter")
    if weather in ["summer", "hot"]:
        styles.append("summer")

    # indoor/outdoor influence
    if scene_type == "indoor" and "formal" not in styles:
        styles.append("indoor")
    elif scene_type == "outdoor":
        styles.append("outdoor")

    return styles

# Embed text tags for similarity search
def encode_texts(texts: List[str]) -> torch.Tensor:
    with torch.no_grad():
        text_tokens = clip.tokenize(texts).to(device)
        text_embeddings = model.encode_text(text_tokens)
        return text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)

# Find outfit images matching styles & tags using CLIP similarity
def find_best_outfits(styles: List[str], top_k=3) -> List[Dict]:
    # Compose search queries from styles
    queries = [style for style in styles]
    text_features = encode_texts(queries)

    results = []

    for outfit in OUTFIT_DB:
        img = preprocess(Image.open(outfit["image_path"])).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(img)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        # Compute similarity to each style embedding, take max
        sims = (text_features @ image_features.T).squeeze(1).cpu().numpy()
        max_sim = sims.max()
        results.append({"outfit": outfit, "score": float(max_sim)})

    # Sort by highest similarity score
    results.sort(key=lambda x: x["score"], reverse=True)

    # Return top-k outfits
    return results[:top_k]

# Generate final recommendation
def recommend_outfits(scene_type: str, occasion: str, weather: str = None):
    styles = suggest_styles(scene_type, occasion, weather)
    color_palette = []
    for style in styles:
        color_palette.extend(COLOR_PALETTES.get(style, []))
    color_palette = list(set(color_palette))  # unique colors

    top_outfits = find_best_outfits(styles)

    recommendations = []
    for item in top_outfits:
        outfit = item["outfit"]
        recommendations.append({
            "image_path": outfit["image_path"],
            "tags": outfit["tags"],
            "score": item["score"],
        })

    return {
        "outfit_recommendation": {
            "styles": styles,
            "color_palette": color_palette
        },
        "outfit_images": recommendations
    }


if __name__ == "__main__":
    # Example usage
    scene = "outdoor"
    occasion = "casual"
    weather = "summer"

    result = recommend_outfits(scene, occasion, weather)

    print("Suggested Styles:", result["outfit_recommendation"]["styles"])
    print("Suggested Color Palette:", result["outfit_recommendation"]["color_palette"])
    print("Top Outfit Images:")
    for outfit in result["outfit_images"]:
        print(f" - {outfit['image_path']} | Tags: {outfit['tags']} | Score: {outfit['score']:.3f}")
