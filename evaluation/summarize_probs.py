import pandas as pd
import numpy as np
import ast
from tqdm import tqdm
import sys
from collections import Counter
import pdb
import argparse
import os
import ast
import scipy.stats as stats

if __name__ == '__main__':

    """
    Run this after running main.py to output a csv file with a table consisting of the following information:

    Columns:
    filepaths,occupation,model,race,probs,top_pred

    `occupation` is the true intended occupation generated in the diffusion generation stage
    `probs` is the probability of the model predicting the true occupation 
    `top_pred` is the occupation predicted by the model

    Example row:
    datasets/final/SDXL_pilot/100008053/Caucasian.jpg,pilot,clip,caucasian,0.428038566031492,flight steward

    Example: python evaluation/summarize_probs.py  outputs/trials/pilot_clip.csv difficult
    """
 
    parser = argparse.ArgumentParser(description = 'eval')
    parser.add_argument('input_csv')
    parser.add_argument('--labels', choices=['difficult', 'base'], help='Specify either `difficult` or `base` as the label choices')
    args = parser.parse_args()

    input_csv = getattr(args, 'input_csv')

    print(f"Evaluating {input_csv}")

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

    # Load CSV
    df = pd.read_csv(input_csv)

    df.race = [x.lower() for x in df.race]

    #Filter out those that do not have a complete set
    counts_df = df.groupby(["prompt_index"])['img_path'].count()
    counts_df = counts_df[counts_df!=5]
    filterout = counts_df.index.tolist()
    df = df[~df.prompt_index.isin(filterout)].reset_index(drop = True)

    occupations = OCCUPATIONS
    # REPRESENTATIONAL FAIRNESS TESTS

    # Create dictionaries to count occupational labels for each race
    base_dict = {}
    black_dict = {}
    caucasian_dict = {}
    asian_dict = {}
    indian_dict = {}

    base_probs = []
    black_probs = []
    caucasian_probs = []
    asian_probs = []
    indian_probs = []

    #Filepaths
    base_fp = []
    black_fp = []
    caucasian_fp = []
    asian_fp = []
    indian_fp = []

    #Top predictions
    base_tp = []
    black_tp = []
    caucasian_tp = []
    asian_tp = []
    indian_tp = []

    right_occupation = df["occupation"].iloc[0]
    #right_occupation = 'a trustworthy person'
    
    # Create key-value pairs and set them to 0
    races = ["caucasian", "black", "asian", "indian", "base"]

    for race in races:
        for occupation in occupations[right_occupation]:
            globals()[race.replace(" ", "_") + "_dict"][occupation] = 0

    for i in tqdm(range(len(df))):
        labels = ast.literal_eval(df["labels"][i])
        race = df["race"][i]
        probs = ast.literal_eval(df["probs"][i])

        #Arrange probs and labels in descending order

        pl = zip(probs, labels)
        pl = sorted(pl, key = lambda x: -x[0])
        probs, labels = zip(*pl)

        globals()[race.replace(" ", "_") + "_fp"].append(df["img_path"][i])
        globals()[race.replace(" ", "_") + "_tp"].append(labels[0])

        globals()[race.replace(" ", "_") + "_dict"][labels[0]] += 1

        for l, p in zip(labels, probs):
            if l == right_occupation:
                globals()[race.replace(" ", "_") + "_probs"].append(p)
        
    # Print out summary results
    # Summary Statistics by Race
    raw_df = pd.DataFrame()
    directory = "/".join(input_csv.split("/")[:-1])
    model = input_csv.split("/")[-1].split("_")[-1].split(".")[0]

    for race in races:
        race_df = df[df["race"] == race]

        int_df = pd.DataFrame.from_records(globals()[race.replace(" ", "_") + "_dict"], index=['counts']).transpose().sort_values('counts', ascending = False)
        probs = globals()[race.replace(' ', '_') + '_probs']
        average_prob = np.array(probs).mean()

        l = len(probs)
        raw_df = pd.concat([raw_df, pd.DataFrame({"filepaths": globals()[race.replace(" ", "_") + "_fp"], "occupation": [right_occupation]*l, "model": [model]*l, "race": [race]*l, "probs": probs, "top_pred": globals()[race.replace(" ", "_") + "_tp"]})])

    raw_df.to_csv(os.path.join(directory, "summary", f"summarized_{right_occupation}_{model}.csv"), index = False)