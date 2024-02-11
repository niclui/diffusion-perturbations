'''
    This script generates masks for a set of images.
    The masks are generated using the GroundingDINO (Liu et al, 2023) and Segment Anything (Kirillov et al, 2023) models.
    First, we use GroundingDINO to generate bounding boxes for the subject of interest using a "person" prompt.
    Second, we use Segment Anything to generate masks for the subject of interest using the bounding boxes from GroundingDINO.

    python mask_generation.py
        --base_folder 'datasets/post_VQA/base' # Folder where you have your filtered base images saved
        --mask_folder 'datasets/post_VQA/mask' # Folder where you will save your masks  
              
    This code is adapted from Grounded-Segment-Anything by IDEA-Research:
    https://github.com/IDEA-Research/Grounded-Segment-Anything

'''

import os
HOME = os.getcwd()
print("HOME:", HOME)
import cv2
import supervision as sv
from typing import List
import torch
import cv2
from tqdm import tqdm
import math
from PIL import Image
import numpy as np
from segment_anything import SamPredictor
import argparse

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load GroundingDINO
from GroundingDINO.groundingdino.util.inference import Model
GROUNDING_DINO_CONFIG_PATH = os.path.join(HOME, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(HOME, "weights", "groundingdino_swint_ogc.pth")
grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)

# Load SAM
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
from segment_anything import sam_model_registry, SamPredictor
sam = sam_model_registry[SAM_ENCODER_VERSION](checkpoint=SAM_CHECKPOINT_PATH).to(device=DEVICE)
sam_predictor = SamPredictor(sam)


def enhance_class_name(class_names: List[str]) -> List[str]:
    return [
        f"all {class_name}s"
        for class_name
        in class_names
    ]

def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)

def main():
    parser = argparse.ArgumentParser(description='Mask Generation')
    parser.add_argument('--mask_folder', type=str, help='folder where you want to save your masks')
    parser.add_argument('--base_folder', type=str, help='directory of images')
    MASK_FOLDER = parser.parse_args().mask_folder
    IMAGES_DIRECTORY = parser.parse_args().base_folder
    
    if not os.path.exists(MASK_FOLDER):
        os.makedirs(MASK_FOLDER)
    IMAGES_EXTENSIONS = ['jpg', 'jpeg', 'png']

    CLASSES = ['person']
    BOX_TRESHOLD = 0.35
    TEXT_TRESHOLD = 0.25

    images = {}
    annotations = {}

    image_paths = sv.list_files_with_extensions(
        directory=IMAGES_DIRECTORY, 
        extensions=IMAGES_EXTENSIONS)

    for image_path in tqdm(image_paths):
        image_name = image_path.name
        image_path = str(image_path)
        image = cv2.imread(image_path)

        detections = grounding_dino_model.predict_with_classes(
            image=image,
            classes=enhance_class_name(class_names=CLASSES),
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )
        detections = detections[detections.class_id != None]
        detections.mask = segment(
            sam_predictor=sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )
        images[image_name] = image
        annotations[image_name] = detections
        print(image_name)
        # Save mask image as PIL
        if annotations[image_name].mask.any():
            mask = annotations[image_name].mask[0]
            mask = Image.fromarray(mask)
            mask.save(os.path.join(MASK_FOLDER, "mask_" + image_name))

main()