import os
import torch
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import urllib.request

# Step 1: Load Places365 labels
def load_labels():
    categories_file = 'categories_places365.txt'
    if not os.path.exists(categories_file):
        urllib.request.urlretrieve(
            'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt',
            categories_file)
    with open(categories_file) as f:
        classes = [line.strip().split(' ')[0][3:] for line in f]
    return classes

# Step 2: Indoor/Outdoor labels (manually defined)
IO_LABELS = {
    'indoor': ['library', 'office', 'bedroom', 'kitchen', 'gym', 'mall', 'classroom', 'church indoor', 'living room'],
    'outdoor': ['beach', 'park', 'street', 'forest', 'mountain', 'farm', 'stadium', 'airport terminal', 'bridge']
}

# Step 3: Load the pre-trained model
def load_model():
    model = models.resnet18(num_classes=365)
    model_file = 'resnet18_places365.pth.tar'
    if not os.path.exists(model_file):
        print("Downloading model...")
        model_url = 'http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar'
        urllib.request.urlretrieve(model_url, model_file)
    checkpoint = torch.load(model_file, map_location='cpu')
    state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model

# Step 4: Preprocess the image
def preprocess_image(img_path):
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert('RGB')
    return transform(img).unsqueeze(0)

# Step 5: Classify scene and infer indoor/outdoor
def classify_scene(img_path):
    classes = load_labels()
    model = load_model()
    input_tensor = preprocess_image(img_path)

    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        top5_prob, top5_idx = torch.topk(probs, 5)

    top_scenes = [(classes[i], top5_prob[idx].item()) for idx, i in enumerate(top5_idx)]

    # Decide Indoor/Outdoor from top scene
    top_scene = top_scenes[0][0]
    scene_type = 'Indoor' if any(indoor in top_scene for indoor in IO_LABELS['indoor']) else 'Outdoor'

    return {
        "scene_type": scene_type,
        "location_theme": top_scene,
        "background_features": top_scenes
    }

# -------- RUN THE MODULE --------
if __name__ == "__main__":
    image_path = 'Harini-Aswin-MCC-Hall-Chennai-0609+-+Copy.jpg'  # Replace with your image path
    result = classify_scene(image_path)
    
    print("📸 Scene Classification Result:")
    print(f"Scene Type: {result['scene_type']}")
    print(f"Top Scene: {result['location_theme']}")
    print("Top 5 Predictions:")
    for scene, prob in result['background_features']:
        print(f"  {scene}: {prob:.2f}")
