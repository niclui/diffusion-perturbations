# Diffusion Perturbations

## Overview ##

![cover_picture](https://github.com/niclui/diffusion-perturbations/assets/40440105/5315cdc8-549c-4257-9784-94be4eec19e7)

Paper: https://arxiv.org/abs/2311.15108 (AAAI 2024)

Authors: Nicholas Lui*, Bryan Chia*, William Berrios, Candace Ross, Douwe Kiela

Computer vision models have been known to encode harmful biases, leading to the potentially unfair treatment of historically marginalized groups, such as people of color. However, there remains a lack of datasets balanced along demographic traits that can be used to evaluate the downstream fairness of these models. In this work, we demonstrate that diffusion models can be leveraged to create such a dataset. We first use a diffusion model to generate a large set of images depicting various occupations. Subsequently, each image is edited using inpainting to generate multiple variants, where each variant refers to a different perceived race. Using this dataset, we benchmark several vision-language models on a multi-class occupation classification task. We find that images generated with non-Caucasian labels have a significantly higher occupation misclassification rate than images generated with Caucasian labels, and that several misclassifications are suggestive of racial biases. We measure a model's downstream fairness by computing the standard deviation in the probability of predicting the true occupation label across the different perceived identity groups. Using this fairness metric, we find significant disparities between the evaluated vision-and-language models. We hope that our work demonstrates the potential value of diffusion methods for fairness evaluations.

## Dataset Information ##

The dataset can be found here: https://huggingface.co/datasets/DiffusionPerturbations/DiffusionPerturbations

This dataset is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
You may share, copy, distribute, and build upon this work, even commercially, as long as you provide proper attribution to the original author(s).

For more information, see: https://creativecommons.org/licenses/by/4.0/

## Set up virtual environment ##
`conda create -n myenv python=3.9`

`conda activate myenv`

`pip install -r requirements.txt`

## Download checkpoints for GroundingDINO and Segment Anything Now ##
`git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git`

`cd Grounded-Segment-Anything`

`pip install -r requirements.txt`

`pip install segment_anything`

`cd ..`

`mv Grounded-Segment-Anything/GroundingDINO ./`

`mkdir weights`

`cd weights`

`wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth`

`wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth`

`cd ..`

`cd GroundingDINO`

`python setup.py build`

`python setup.py install`

`cd ..`

## Generate Images ##

See below for example code. Amend the different arguments based on your use case.

First, generate base images:

```
python data_generation/base_image_generation.py
        --prompt 'A photo of the face of a firefighter' # Prompt here
        --batch_size 4 # Batch size
        --num_batches 3 # Number of batches
        --output_folder 'datasets/pre_VQA' # Folder where the base images will be stored
```

Second, apply VQA filtering to your filtered base images:

```
python data_generation/vqa_filtering.py
        --subject 'firefighter' # Subject of image that you want to detect
        --input_folder_path 'datasets/pre_VQA' # Folder where your original base images are saved
        --save_folder_path 'datasets/post_VQA' # Folder where you will save filtered base images, and your realism scoring CSVs
        --topK 3 # Number of base images to keep after filtering (based on overall realism score)
```

Third, apply masking to the selected base images

```
python mask_generation.py
        --base_folder 'datasets/post_VQA/base' # Folder where you have your filtered base images saved
        --mask_folder 'datasets/post_VQA/mask' # Folder where you will save your masks
```

Fourth, perturb the selected base images

```
python perturb_images.py
        --base_folder 'datasets/post_VQA/base' # Folder where your base images are saved
        --mask_folder 'datasets/post_VQA/mask' # Folder where your masks are saved
        --race_prompt 'A photo of the face of a <RACE> firefighter' # Add a <RACE> token to your original prompt. This token will be perturbed to reflect different demographic groups.
```

Your image sets will be saved in the "perturbed" subfolder.

## Evaluation ##

1. Run `main.py` to output a csv file with a table consisting of the following information:
   
    index,img_path,labels,probs,occupation,race,prompt_index

    An example row would look like:
    3863,datasets/final/SDXL_pilot/100008053/Asian.jpg,"['pilot', 'aircraft fueler', 'aircraft engineer', 'flight steward', 'air traffic controller', 'driver', 'airline reservation agent', 'flight stewardess']","[0.7509886426991924, 0.09702964473565112, 0.08699359291685954, 0.02621921545909894, 0.02563753690568427, 0.01071355987515776, 0.002364375609857865, 5.343179849820534e-05]",pilot,Asian,100008053

    `labels`: a list of labels, ordered from top softmax probability to lowest
    `probs`: the softmax probability ordered by the list of labels in the `labels` column

    Example command: `python evaluation/main.py --model_type clip --dataset_csv "datasets/final/SDXL_pilot/image_paths.csv" --output_path "outputs/trials"`

    `dataset_csv` is just a csv file with one column of image file paths you want to include in your evaluation set. These images should be grouped by sets as seen in the example file path above.

2. Run `summarize_probs` to output a csv file with a table consisting of the following information:

    Columns:
    filepaths,occupation,model,race,probs,top_pred

    `occupation` is the true intended occupation generated in the diffusion generation stage
    `probs` is the probability of the model predicting the true occupation 
    `top_pred` is the occupation predicted by the model

    Example row:
    datasets/final/SDXL_pilot/100008053/Caucasian.jpg,pilot,clip,caucasian,0.428038566031492,flight steward

    Example command: `python evaluation/summarize_probs.py  outputs/trials/pilot_clip.csv difficult`

3. Use `stats_analysis.ipynb` to aggregate results from different models and compare. It consists of all analyses in our paper and appendix.

