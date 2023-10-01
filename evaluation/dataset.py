import numpy as np
import pandas as pd
import torch
from PIL import Image
import re
from glob import glob
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

class FairnessDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_csv, model, occupations, preprocess = None):

        RACES = ["Caucasian", "Black", "Asian", "Indian"]
        
        self._df = pd.read_csv(dataset_csv)
        self.model = model
        self.preprocess = preprocess  

        self._df = self._df[["_noCF_" not in x for x in self._df["img_path"]]].reset_index(drop = True)
        occupation_str = "|".join(list(occupations.keys()))
        self._df['occupation'] = [re.search(f"({occupation_str})", x).group(1) for x in self._df['img_path']]
        race_str = "|".join(RACES) + ("|base")
        self._df['race'] = [re.search(f"({race_str})", x).group(1) for x in self._df['img_path']]
        self._occupations = self._df["occupation"]
        self._race = self._df["race"]

        self._image_path = self._df["img_path"]

    def __len__(self):
        return len(self._df)

    def __getitem__(self, index):
        img_path = self._image_path[index]

        if self.model == "clip":
            raw_image = Image.open(img_path)
            image = self.preprocess(raw_image)
        elif self.model == "flava":
            raw_image = Image.open(img_path)
            image = self.preprocess(images=raw_image, return_tensors="pt")
            image["pixel_values"] = image["pixel_values"].squeeze()
        elif self.model[:10] == "laion-clip":
            raw_image = Image.open(img_path)
            image = self.preprocess(raw_image)

        return img_path, image