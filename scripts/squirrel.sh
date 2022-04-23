#!/bin/bash

python train.py \
    --dataset squirrel \
    --dropout 0.1 \
    --lr 0.001 \
    --hidden 64\
    --num_layer 2\
    --weight_decay 5e-5\
    --epochs 1000 \
    --split_id 0