'''
    This script generates base images for a given prompt using a diffusion model.

    Example usage:
        python base_image_generation.py
        --prompt 'A photo of the face of a firefighter'
        --batch_size 4
        --num_batches 250
        --output_folder 'datasets/base_images'
'''
import numpy as np
import torch
import argparse
import os
from diffusers import DiffusionPipeline, StableDiffusionXLInpaintPipeline

# Load Stable Diffusion XL pipeline
pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipe.to(device)
pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

def main():
    parser = argparse.ArgumentParser(description='Generate base images with Stable Diffusion XL')
    parser.add_argument('--prompt', type=str, default="A photo of the face of a firefighter", help='prompt to use')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--num_batches', type=int, default=250, help='number of batches')
    parser.add_argument('--output_folder', type=str, default='datasets/base_images', help='output folder')

    prompt = parser.parse_args().prompt
    batch_size = parser.parse_args().batch_size
    num_batches = parser.parse_args().num_batches
    print(f"Prompt: {prompt}, Batch Size: {batch_size}, Total Images Generating: {batch_size * num_batches}")

    # Create folder path
    fp = parser.parse_args().output_folder
    if not os.path.exists(fp):
        os.makedirs(fp)

    # Begin generation
    negative_prompts = "" # TODO: Add your negative prompts here
    for i in range(num_batches):
        seeds = np.arange(i*batch_size, (i+1)*batch_size)
        torch_generators = [torch.Generator().manual_seed(int(seed)) for seed in seeds]
        images = pipe(prompt=[prompt] * batch_size,
                    negative_prompt = [negative_prompts] * batch_size,
                    num_images_per_prompt=1,
                    generator=torch_generators,
                    ).images

        # Save base images
        for i in range(len(images)):
            image = images[i]
            image.save(os.path.join(fp, f"{str(seeds[i])}_base.jpg"))

    print("IMAGE GENERATION COMPLETE")

main()
    