# diffusion-perturbations

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
