#!/bin/bash

#  for shot in 2 8 16 32; 
#  do for seed in 1 2 3 4 5 6 7 8 9 10; 
 
for shot in 2 32; do
    for seed in {1..3}; do
        python CMA_fewshot.py \
        --seed $seed \
        --dataset_name 'politifact' \
        --train_path "D:/Datasets/DGM4/metadata/train.json" \
        --test_path "D:/Datasets/DGM4/metadata/test.json" \
        --img_path "D:/Datasets/FakenewsData/poli_img_all/" \
        --shot $shot \
        --save_path "./saved_adapter"
    done
done

