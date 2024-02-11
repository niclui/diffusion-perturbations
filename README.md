# Diffusion Perturbations

## Overview ##

![cover_picture](https://github.com/niclui/diffusion-perturbations/assets/40440105/5315cdc8-549c-4257-9784-94be4eec19e7)

Paper: https://arxiv.org/abs/2311.15108 (AAAI 2024)

Authors: Nicholas Lui*, Bryan Chia*, William Berrios, Candace Ross, Douwe Kiela

Computer vision models have been known to encode harmful biases, leading to the potentially unfair treatment of historically marginalized groups, such as people of color. However, there remains a lack of datasets balanced along demographic traits that can be used to evaluate the downstream fairness of these models. In this work, we demonstrate that diffusion models can be leveraged to create such a dataset. We first use a diffusion model to generate a large set of images depicting various occupations. Subsequently, each image is edited using inpainting to generate multiple variants, where each variant refers to a different perceived race. Using this dataset, we benchmark several vision-language models on a multi-class occupation classification task. We find that images generated with non-Caucasian labels have a significantly higher occupation misclassification rate than images generated with Caucasian labels, and that several misclassifications are suggestive of racial biases. We measure a model's downstream fairness by computing the standard deviation in the probability of predicting the true occupation label across the different perceived identity groups. Using this fairness metric, we find significant disparities between the evaluated vision-and-language models. We hope that our work demonstrates the potential value of diffusion methods for fairness evaluations.

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

See below for example code. Amend the different argumetns based on your use case.

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
        --race_prompt 'A photo of the face of a &lt;RACE&gt; firefighter' # Add a &lt;RACE&gt; token to your original prompt. This token will be perturbed.
```

Your image sets will be saved in the "perturbed" subfolder.
