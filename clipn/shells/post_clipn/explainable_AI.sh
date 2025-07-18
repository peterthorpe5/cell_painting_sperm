#!/bin/bash




cd /home/pthorpe/apps/cell_painting_sperm/


git pull 

wait 


cd /home/pthorpe/scratch/explainable/STB_vs_mitotox_reference_only_E100_L20/post_clipn/post_analysis_script

python "$HOME/apps/cell_painting_sperm/clipn/identify_acrosome_abnormalities.py"         --ungrouped_list    \
     /mnt/shared/projects/uosa/Pete_Thorpe/2025_cell_painting_sperm/image_level_data/datasets/dataset_M_S_S_Image_level.csv




python $HOME/apps/cell_painting_sperm/clipn/explain_feature_driven_results.py --output_dir nn_with_n \
     --ungrouped_list      \   
     /mnt/shared/projects/uosa/Pete_Thorpe/2025_cell_painting_sperm/image_level_data/datasets/dataset_M_S_S_Image_level.csv \
    --nn_file nearest_neighbours.tsv



# using median per cpd_id - not enough data!!
python $HOME/apps/cell_painting_sperm/clipn/shap_explain_nn_similarity.py --nn_file nearest_neighbours.tsv  \
    --features   /mnt/shared/projects/uosa/Pete_Thorpe/2025_cell_painting_sperm/image_level_data/datasets/dataset_M_S_S_Image_level.csv  \
    --output_dir  shap_analysis_real  \
    --n_neighbors 10
