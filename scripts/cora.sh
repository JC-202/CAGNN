#!/bin/bash

python train.py \
    --dataset cora \
    --dropout 0.5 \
    --lr 0.05 \
    --hidden 64\
    --num_layer 2\
    --weight_decay 5e-4\
    --epochs 1000 \
    --split_id 0 \
