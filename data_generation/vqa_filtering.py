'''
    Using a VQA model, this script filters a set of base images based on three categories:
    1. Text-to-image faithfulness (does this image contain the subject of interest that I was trying to depict?)
    2. Limb realism (are the limbs of the subject of interest distorted?)
    3. Overall realism (is this image real or fake?)
    
    We do an additional check to filter grayscale images by computing the # of unique colors present:
    4. Grayscale check (is this image grayscale?) 

    First, we eliminate base images that fail any of these checks.
    Second, we rank the remaining base images based on their overall realism score (based on Q3 above)
    Third, we keep the top K base images based on this ranking.

    The script outputs two CSV files:
    1. vqa_results.csv: contains the results of the VQA and grayscale checks for each base image
    2. base_filtered_images.csv: contains the paths to the final filtered set of base images
    
    Example usage:
    python data_generation/vqa_filtering.py
        --subject 'firefighter' # Subject of image that you want to detect
        --input_folder_path 'datasets/pre_VQA' # Folder where your original base images are saved
        --save_folder_path 'datasets/post_VQA' # Folder where you will save filtered base images, and your realism scoring CSVs
        --topK 5 # Number of base images to keep after filtering (based on overall realism score)
'''

from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image
import os
import glob
import pandas as pd
import cv2
from tqdm import tqdm
import torch
import argparse

# Load ViLT-B/32 VQA model
processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = model.to(device)

# Compute number of unique colors to decide if an image is grayscale or not
def unique_colors(image_path):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    unique_colors = len(set(tuple(color) for color in rgb_image.reshape(-1, 3)))
    return unique_colors

def main():
    parser = argparse.ArgumentParser(description='Choose')
    parser.add_argument('--subject', type=str, help='subject of interest')
    parser.add_argument('--input_folder_path', type=str, help='input folder path containing all the original base images')
    parser.add_argument('--save_folder_path', type=str, help='folder path where you want to save your CSV results and filtered images')
    parser.add_argument('--topK', type=int, help='topK base images to keep based on VQA overall realism score')

    subject_of_interest = parser.parse_args().subject
    input_folder_path = parser.parse_args().input_folder_path
    save_folder_path = parser.parse_args().save_folder_path
    topK = parser.parse_args().topK

    # check if input folder of images exist:
    if not os.path.exists(input_folder_path):
        print(f"Folder path {input_folder_path} does not exist")
        exit()

    # Check if save folder exists. if not, create it
    if not os.path.exists(save_folder_path):
        os.makedirs(save_folder_path)
    # Create folder for filtered images
    if not os.path.exists(os.path.join(save_folder_path, "base")):
        os.makedirs(os.path.join(save_folder_path, "base"))
    
    # Generate a CSV of image paths for images in the input_folder_path
    folder_path = os.path.join(input_folder_path, "image_paths.csv")
    image_paths = glob.glob(os.path.join(input_folder_path, "*.jpg"))
    df = pd.DataFrame({"img_path": image_paths})
    df.to_csv(folder_path)

    # Now, we will run the VQA model on the images in the input_folder_path
    df = pd.read_csv(folder_path)
    filenames = df.img_path.tolist()

    # TO-DO: Batch the images to speed up the process
    # VQA Question 1: Text-to-image faithfulness
    question = f"Is there a {subject_of_interest} in this image?"
    txt2img_faith = []
    for fp in tqdm(filenames):
        # prepare inputs
        image = Image.open(fp)
        encoding = processor(image, question, return_tensors="pt")
        encoding.to("cuda")
        outputs = model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        if model.config.id2label[idx] == "yes":
            txt2img_faith.append(1)
        else:
            txt2img_faith.append(0)

    # VQA Question 2: Limb realism
    question = f"Are this person's limbs distorted?"
    limb_distorted = []
    for fp in tqdm(filenames):
        # prepare inputs
        image = Image.open(fp)
        encoding = processor(image, question, return_tensors="pt")
        encoding.to("cuda")
        outputs = model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        if model.config.id2label[idx] == "yes":
            limb_distorted.append(1)
        else:
            limb_distorted.append(0)

    # VQA Question 3: Overall realism
    question = "Is this image real or fake?"
    image_fake = []
    realism_score = []
    for fp in tqdm(filenames):
        # prepare inputs
        image = Image.open(fp)
        encoding = processor(image, question, return_tensors="pt")
        encoding.to("cuda")
        outputs = model(**encoding)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        if model.config.id2label[idx] == "fake": #1812
            image_fake.append(1)
        else: #142
            image_fake.append(0)
        realism_score.append(logits[0][142].item()) # this is my VQA overall realism score

    # Grayscale Check
    unique_color_list = []
    for fp in tqdm(filenames):
        color = unique_colors(fp)
        unique_color_list.append(color)

    results_df = pd.DataFrame({"img_path": filenames,
                               "txt2img_faith": txt2img_faith,
                               "limb_distorted": limb_distorted,
                               "image_fake": image_fake,
                               "realism_score": realism_score,
                               "unique_colors": unique_color_list})
    print(len(results_df))
    results_df.to_csv(os.path.join(save_folder_path, "vqa_results.csv"))

    # Finally, filter the images
    greyscale_threshold = 10000 # Increase this threshold if you want more colorful images
    print(f"Total number of images: {len(results_df)}")
    print(f"Based on VQA, # of images that contain {subject_of_interest}: {len(results_df[results_df['txt2img_faith']==1])}")
    print(f"Based on VQA, # of images that have limb distortion: {len(results_df[results_df['limb_distorted']==1])}")
    print(f"# of images that are # of unique colors that exceed greyscale_threshold = {greyscale_threshold}: \
            {len(results_df[results_df['unique_colors']>=greyscale_threshold])}")

    # Filter the images based on the VQA questions
    filtered_df = results_df.copy()
    filtered_df = filtered_df[filtered_df['txt2img_faith']==1]
    filtered_df = filtered_df[filtered_df['limb_distorted']==0]
    filtered_df = filtered_df[filtered_df['unique_colors']>=greyscale_threshold]

    # Finally, take the topK images based on the realism score
    filtered_df = filtered_df.sort_values(by=['realism_score'], ascending=False)
    filtered_df = filtered_df.head(topK)
    filtered_df.to_csv(os.path.join(save_folder_path, "filtered_base_images.csv"))
    filtered_df = pd.read_csv(os.path.join(save_folder_path, "filtered_base_images.csv"))
    
    # Save all the images in the filtered_df
    for idx, row in filtered_df.iterrows():
        img_path = row['img_path']
        img = Image.open(img_path)
        index = img_path.split("/")[-1].split(".")[0].split("_")[0]
        img.save(os.path.join(os.path.join(save_folder_path, "base"), f"{index}_filtered.jpg"))

    print("VQA AND GREYSCALE FILTERING COMPLETE")

main()