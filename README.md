# CAGNN
A **PyTorch** implementation of CAGNN **"Exploiting Neighbor Effect: Conv-Agnostic GNNs Framework for Graphs with Heterophily"**.


## Dependencies
- python 3.7.3
- pytorch 1.6.0
- dgl 0.6.0
- torch-geometric 1.6.2

## Code Architecture
    |── datasets                # datasets and load scripts
    |── utils                   # Common useful modules(transform, loss function)
    |── models                  # models 
    |  └── layers               # code for layers
    |  └── models               # code for models
    |── scripts                 # train scripts for each dataset     
    └── train.py                # basic trainner and hyper-parameter
    

## Train 

```
python train.py
```

## Scripts
```
sh scripts/texas.sh
sh scripts/wisconsin.sh
sh scripts/actor.sh
sh scripts/squirrel.sh
sh scripts/chameleon.sh
sh scripts/cornell.sh
sh scripts/citeseer.sh
sh scripts/pubmed.sh
sh scripts/cora.sh
```