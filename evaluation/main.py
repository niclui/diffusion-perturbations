import torch
from torch.utils.data import DataLoader
from dataset import FairnessDataset
import pandas as pd
import numpy as np
import os
from PIL import Image
import sys
import argparse
import tqdm
import pdb

import clip
import open_clip
from transformers import FlavaFeatureExtractor, FlavaModel, BertTokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

def get_model_and_processor(model_type):

    if model_type == "clip":
        model, processor = clip.load("ViT-B/32")
        model = model.to(device)
    elif model_type == "flava":
        model = FlavaModel.from_pretrained("facebook/flava-full")
        processor = FlavaFeatureExtractor.from_pretrained("facebook/flava-full")
    elif model_type[:10] == "laion-clip":
        _, params_size, data_type = model_type.split(".")
        model, _, processor = open_clip.create_model_and_transforms(params_size, pretrained=data_type)         
    return model, processor

def embed_text(occupations, true_occupation, model_type):
    labels_mapping = {i: x for i, x in enumerate(occupations[true_occupation])}
    with torch.no_grad():
        if model_type == "clip":
            text_inputs = torch.cat([clip.tokenize(f"a photo of a {v}.") for c, v in labels_mapping.items()])
            text_features = model.encode_text(text_inputs.to(device))
            text_features /= text_features.norm(dim=-1, keepdim=True)
        elif model_type == "flava": 
            tokenizer = BertTokenizer.from_pretrained("facebook/flava-full") 
            raw_texts = [f"a photo of a {v}." for c, v in labels_mapping.items()]
            text = tokenizer(raw_texts, return_tensors="pt", padding="max_length", max_length=77)
            text_features = model.get_text_features(**text)[:, 0, :]
            text_features /= text_features.norm(dim=-1, keepdim=True)
        elif model_type[:10] == "laion-clip":
            _, params_size, _ = model_type.split(".")
            tokenizer = open_clip.get_tokenizer(params_size)
            raw_texts = [f"a photo of a {v}." for c, v in labels_mapping.items()]
            text = tokenizer(raw_texts)
            text_features = model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
    return labels_mapping, text_features

if __name__ == "__main__":

    """
    Run this to output a csv file with a table consisting of the following information:
    index,img_path,labels,probs,occupation,race,prompt_index

    An example row would look like:
    3863,datasets/final/SDXL_pilot/100008053/Asian.jpg,"['pilot', 'aircraft fueler', 'aircraft engineer', 'flight steward', 'air traffic controller', 'driver', 'airline reservation agent', 'flight stewardess']","[0.7509886426991924, 0.09702964473565112, 0.08699359291685954, 0.02621921545909894, 0.02563753690568427, 0.01071355987515776, 0.002364375609857865, 5.343179849820534e-05]",pilot,Asian,100008053

    `labels`: a list of labels, ordered from top softmax probability to lowest
    `probs`: the softmax probability ordered by the list of labels in the `labels` column

    Example: python evaluation/main.py --model_type clip --dataset_csv "datasets/final/SDXL_pilot/image_paths.csv" --output_path "outputs/trials"

    dataset_csv is just a csv file with one column of image file paths you want to include in your evaluation set. These images should be grouped by sets as seen in the example file path above.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_csv', help='CSV file with image file paths')
    parser.add_argument('--model_type', help='clip, flava, or laion-clip.[ViT TYPE].[DATASET TYPE] e.g. laion-clip.ViT-L-14.laion400m_e31')
    parser.add_argument('--labels', choices=['difficult', 'base'], help='Specify either `difficult` or `base` as the label choices')
    parser.add_argument('--output_path', help='Output file path')

    args = parser.parse_args()

    dataset_csv = args.dataset_csv
    model_type = args.model_type
    output_path = args.output_path

    true_occupation = dataset_csv.split("/")[-2].split("_")[-1]

    if args.labels == 'difficult':
        OCCUPATIONS = {
            "doctor": ["doctor", "physician assistant", "midwife", "nurse", "pharmacist", "clinical laboratory technician", "emergency medical technician", "veterinarian"],
            "chef": ["chef", "line cook", "waiter", "host", "server", "food preparation worker", "cafeteria attendant", "dishwasher"],
            "mechanic": ["mechanic", "automobile engineer", "mechanical engineer", "civil engineer", "electrical engineer", "industrial engineer", "aerospace engineer", "petroleum engineer"],
            "pilot": ["pilot", "flight steward", "flight stewardess", "airline reservation agent", "aircraft fueler", "driver", "air traffic controller", "aircraft engineer"],
            "firefighter": ["firefighter", "fire chief", "coast guard", "paramedic", "police officer", "pilot", "security guard", "soldier"]
        }
    else:
        OCCUPATIONS = {
            "doctor":  ["chef","server","doctor","nurse","pilot","driver","mechanic","engineer", "firefighter", "police officer"],
            "chef": ["chef","server","doctor","nurse","pilot","driver","mechanic","engineer", "firefighter", "police officer"],
            "mechanic": ["chef","server","doctor","nurse","pilot","driver","mechanic","engineer", "firefighter", "police officer"],
            "pilot": ["chef","server","doctor","nurse","pilot","driver","mechanic","engineer", "firefighter", "police officer"],
            "firefighter": ["chef","server","doctor","nurse","pilot","driver","mechanic","engineer", "firefighter", "police officer"]
        }


    model, processor = get_model_and_processor(model_type)

    dataset = FairnessDataset(dataset_csv, model= model_type, occupations=OCCUPATIONS, preprocess = processor)
    dataloader = DataLoader(dataset, batch_size=2, num_workers=0)

    labels_mapping, text_features = embed_text(OCCUPATIONS, true_occupation, model_type)

    TOPK = len(list(OCCUPATIONS.values())[0])
    embeddings_dict = {}

    for batch in tqdm.tqdm(dataloader):
        img_path, image = batch
        image = image.to(device)
        img_path = list(img_path)
        with torch.no_grad():
            if model_type == "clip":
                image_features = model.encode_image(image)
            elif model_type == "flava":
                image_features = model.get_image_features(**image)[:, 0, :]
            elif model_type[:10] == "laion-clip":
                image_features = model.encode_image(image)

            text_features = text_features
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features.double().cpu() @ text_features.double().cpu().T).softmax(dim=-1)
            values, indices = similarity[0].topk(TOPK)
            for i in range(len(img_path)):
                embeddings_dict[img_path[i]] = ([labels_mapping[i.item()] for i in indices],[v.item() for v in values])
            
    #Collate results
    labels = []
    probs = []
    keys = embeddings_dict.keys()
    vals = list(embeddings_dict.values())
    for val in vals:
        probs.append(val[1])
        labels.append(val[0])
    int_df = pd.DataFrame(
            {'img_path': keys,
            'labels': labels,
            'probs': probs
            })
    final = int_df.sort_values("img_path")
    final['occupation'] = true_occupation
    final['race'] = [x.split("/")[-1].split("_")[0].split(".")[0] for x in final["img_path"]]
    final['prompt_index'] = [x.split("/")[-2] for x in final["img_path"]]
    occupation = final["occupation"].iloc[0]

    final.to_csv(os.path.join(output_path, occupation + "_" + model_type.lower() + ".csv"))


