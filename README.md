# diffusion-perturbations
conda create -n test python=3.9

# Download checkpoints for GroundingDINO and Segment Anything Now
`git clone https://github.com/IDEA-Research/Grounded-Segment-Anything.git`

`cd Grounded-Segment-Anything`

`pip install -r 'requirements.txt'`

`pip install segment_anything`

`mv Grounded-Segment-Anything/GroundingDINO ./`

`mkdir weights`

`cd weights`

`wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth`

`wget https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth`

`pip install -e GroundingDINO`



# Run mask generation

