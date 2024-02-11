'''
    Perturb base images.

    python perturb_images.py
        --base_folder 'datasets/post_VQA/base' # Folder where your base images are saved
        --mask_folder 'datasets/post_VQA/mask' # Folder where your masks are saved
        --race_prompt 'A photo of the face of a <RACE> firefighter' # Add a <RACE> token to your original prompt. The token will be perturbed to different demographic groups.
'''
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from diffusers import StableDiffusionXLInpaintPipeline, DiffusionPipeline
from diffusers.utils import load_image
import pandas as pd
from PIL import Image
from tqdm import tqdm
import argparse

# Inpainting pipeline
pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipe.to("cuda")

# Refiner
refiner = DiffusionPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-refiner-1.0",
    text_encoder_2=pipe.text_encoder_2,
    vae=pipe.vae,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
refiner.to("cuda")

def main():
    # Parse Arguments
    parser = argparse.ArgumentParser(description='SDXL Inpainting')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--base_folder', type=str, help='folder path of filtered base images')
    parser.add_argument('--mask_folder', type=str, help='folder path of masks')
    parser.add_argument('--race_prompt', type=str, help='prompt used to generate the image with a <RACE> tokens')

    batch_size = parser.parse_args().batch_size
    base_folder = parser.parse_args().base_folder
    mask_folder = parser.parse_args().mask_folder
    race_prompt = parser.parse_args().race_prompt

    image_paths_list = [os.path.join(base_folder, x) for x in os.listdir(base_folder)]
    mask_paths_list = [os.path.join(mask_folder, "mask_" + x) for x in os.listdir(base_folder)]

    # Make sure all the masks exist
    for mask_path in mask_paths_list:
        if not os.path.exists(mask_path):
            raise ValueError("Mask does not exist")
    print("Verified that all masks exist")

    race_prompts = [race_prompt] * len(image_paths_list)

    # Negative prompts
    negative_prompts = "" # TO-DO: Add negative prompts to guide image inpainting here

    # Auxiliary Functions
    def load_and_preprocess_images(image_paths):
        images = []
        for path in image_paths:
            img = Image.open(path)
            images.append(img)
        return images

    # Batching
    def batch_images_and_prompts(image_paths, masks, images, race_prompts, batch_size):
        num_image_paths = len(image_paths)
        num_masks = len(masks)
        num_images = len(images)
        num_prompts = len(race_prompts)

        assert num_image_paths == num_masks == num_images == num_prompts, "Number of images and prompts must be equal"
        
        image_path_batches = [image_paths[i:i + batch_size] for i in range(0, num_images, batch_size)]
        mask_batches = [masks[i:i + batch_size] for i in range(0, num_images, batch_size)]
        image_batches = [images[i:i + batch_size] for i in range(0, num_images, batch_size)]
        prompt_batches = [race_prompts[i:i + batch_size] for i in range(0, num_images, batch_size)]
        return image_path_batches, mask_batches, image_batches, prompt_batches

    # Step 1: Load and preprocess the images
    images = load_and_preprocess_images(image_paths_list)
    masks = load_and_preprocess_images(mask_paths_list)

    # Step 2: Organize the images into batches
    image_path_batches, mask_batches, image_batches, prompt_batches = batch_images_and_prompts(image_paths_list, masks, images, race_prompts, batch_size)

    # Now, you can iterate through the batches and use them with your HuggingFace model
    for i in tqdm(range(len(image_batches))):
        image_path_batch = image_path_batches[i]
        mask_batch = mask_batches[i]
        image_batch = image_batches[i]
        prompt_batch = prompt_batches[i]
        # create a folder for each image
        subfolder_paths = [os.path.join("/".join(x.split("/")[:-2]), "perturbed", image_path_batch[i].split("/")[-1].split("_")[0]) for i, x in enumerate(image_path_batch)]
        for subfolder_path in subfolder_paths:
            if not os.path.exists(subfolder_path):
                os.makedirs(subfolder_path)

        # save initial image
        for j in range(len(image_batch)):
            image_batch[j].save(os.path.join(subfolder_paths[j], "base.jpg"))
            mask_batch[j].save(os.path.join(subfolder_paths[j], "mask.jpg"))

        for race in ["Black", "Caucasian", "Indian", "Asian"]:
            race_prompt_batch = [x.replace("<RACE>", race) for x in prompt_batch]
            for IDX in range(len(image_batch)):
                image=pipe(
                    prompt=race_prompt_batch[IDX],
                    image=image_batch[IDX],
                    mask_image=mask_batch[IDX],
                    num_inference_steps=40,
                    denoising_end=0.8,
                    strength=0.7,
                    output_type="latent",
                ).images

                perturbed_images=refiner(
                    prompt=race_prompt_batch[IDX],
                    num_inference_steps=40,
                    denoising_start=0.8,
                    image=image,
                ).images

                # now, time to save
                perturbed_images[0].save(os.path.join(subfolder_paths[IDX], f"{race}.jpg"))

main()