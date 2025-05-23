import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image
import cv2
import numpy as np

# Load ControlNet pretrained model for pose (openpose or canny edge etc.)
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-openpose", torch_dtype=torch.float16
)

# Load Stable Diffusion pipeline with ControlNet
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16
)
pipe.enable_xformers_memory_efficient_attention()
pipe.to("cuda")

def preprocess_pose_image(pose_image_path):
    # Load pose image (like openpose keypoints visualization)
    image = cv2.imread(pose_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (512, 512))
    return Image.fromarray(image)

def generate_mockup(pose_img_path, prompt, negative_prompt="", num_inference_steps=30, guidance_scale=7.5):
    pose_conditioning = preprocess_pose_image(pose_img_path)
    
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        image=pose_conditioning,
    )
    return output.images[0]

if __name__ == "__main__":
    # Example inputs:
    pose_image_path = "pose_example.png"  # a pose skeleton image
    outfit_description = "a trendy summer dress, pastel colors, light fabric"
    background_description = "sunny park with trees and flowers"
    
    # Combine prompt for stable diffusion
    prompt = f"A full body portrait of a person in {outfit_description} standing in {background_description}, photorealistic, highly detailed"

    # Generate the image
    result_img = generate_mockup(pose_image_path, prompt)
    result_img.save("mockup_output.png")
    print("Mockup image saved as mockup_output.png")
