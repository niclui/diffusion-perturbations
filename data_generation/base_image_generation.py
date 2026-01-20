'''
    This script generates base images for a given prompt using a diffusion model.

    Example usage:
    python data_generation/base_image_generation.py
        --prompt 'A photo of the face of a firefighter' # Prompt here
        --batch_size 4 # Batch size
        --num_batches 3 # Number of batches
        --output_folder 'datasets/pre_VQA' # Folder where the base images will be stored
        --model 'sdxl' # Model to use (default: sdxl)

    Available models:
        - sdxl: Stable Diffusion XL (stabilityai/stable-diffusion-xl-base-1.0)
                Trained on LAION-5B subset, high quality 1024x1024 images
        - sd15: Stable Diffusion 1.5 (stable-diffusion-v1-5/stable-diffusion-v1-5)
                Trained on LAION-Aesthetics v2, 512x512 images, faster inference
        - dreamlike: Dreamlike Diffusion 1.0 (dreamlike-art/dreamlike-diffusion-1.0)
                     SD 1.5 fine-tuned on high quality art, different aesthetic style
        - segmind-vega: Segmind Vega (segmind/Segmind-Vega)
                        Distilled SDXL, 70% smaller, 100% faster, trained on Grit + Midjourney data
'''

import numpy as np
import torch
import argparse
import os
from diffusers import DiffusionPipeline, StableDiffusionPipeline, StableDiffusionXLPipeline

SUPPORTED_MODELS = {
    "sdxl": {
        "model_id": "stabilityai/stable-diffusion-xl-base-1.0",
        "pipeline_class": DiffusionPipeline,
        "use_safetensors": True,
        "variant": "fp16",
        "description": "Stable Diffusion XL - trained on LAION-5B subset"
    },
    "sd15": {
        "model_id": "stable-diffusion-v1-5/stable-diffusion-v1-5",
        "pipeline_class": StableDiffusionPipeline,
        "use_safetensors": True,
        "variant": "fp16",
        "description": "Stable Diffusion 1.5 - trained on LAION-Aesthetics v2"
    },
    "dreamlike": {
        "model_id": "dreamlike-art/dreamlike-diffusion-1.0",
        "pipeline_class": StableDiffusionPipeline,
        "use_safetensors": True,
        "variant": None,
        "description": "Dreamlike Diffusion - SD 1.5 fine-tuned on high quality art"
    },
    "segmind-vega": {
        "model_id": "segmind/Segmind-Vega",
        "pipeline_class": StableDiffusionXLPipeline,
        "use_safetensors": True,
        "variant": "fp16",
        "description": "Segmind Vega - distilled SDXL, trained on Grit + Midjourney data"
    }
}


def load_pipeline(model_name):
    """Load the specified diffusion model pipeline."""
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Model '{model_name}' not supported. Available models: {list(SUPPORTED_MODELS.keys())}")
    
    model_config = SUPPORTED_MODELS[model_name]
    print(f"Loading model: {model_config['description']}")
    
    load_kwargs = {
        "torch_dtype": torch.float16,
        "use_safetensors": model_config["use_safetensors"]
    }
    if model_config["variant"]:
        load_kwargs["variant"] = model_config["variant"]
    
    pipe = model_config["pipeline_class"].from_pretrained(
        model_config["model_id"],
        **load_kwargs
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe.to(device)
    
    if model_name in ["sdxl", "segmind-vega"]:
        pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
    
    return pipe


def main():
    parser = argparse.ArgumentParser(description='Generate base images with diffusion models')
    parser.add_argument('--prompt', type=str, help='prompt to use')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    parser.add_argument('--num_batches', type=int, help='number of batches')
    parser.add_argument('--output_folder', type=str, help='output folder')
    parser.add_argument('--model', type=str, default='sdxl', 
                        choices=list(SUPPORTED_MODELS.keys()),
                        help=f'model to use for generation (default: sdxl). Options: {list(SUPPORTED_MODELS.keys())}')

    args = parser.parse_args()
    prompt = args.prompt
    batch_size = args.batch_size
    num_batches = args.num_batches
    model_name = args.model
    total_images = batch_size * num_batches

    pipe = load_pipeline(model_name)
 
    print(f"Model: {model_name}, Prompt: {prompt}, Batch Size: {batch_size}, Total Images Generating: {total_images}")

    # Create folder path
    fp = args.output_folder
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
    