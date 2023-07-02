# CAGNN
A **PyTorch** implementation of CAGNN **"Exploiting Neighbor Effect: Conv-Agnostic GNNs Framework for Graphs with Heterophily"**. (TNNLS 2023)

(https://arxiv.org/abs/2203.11200)

## Abstract
<p align="justify">
Due to the homophily assumption in graph convolution networks (GNNs), a common consensus in the graph node classification task is that GNNs perform well on homophilic graphs but may fail on heterophilic graphs with many inter-class edges. However, the previous inter-class edges perspective and related homo-ratio metrics cannot well explain the GNNs performance under some heterophilic datasets, which implies that not all the inter-class edges are harmful to GNNs. In this work, we propose a new metric based on von Neumann entropy to re-examine the heterophily problem of GNNs and investigate the feature aggregation of inter-class edges from an entire neighbor identifiable perspective. Moreover, we propose a simple yet effective Conv-Agnostic GNN framework (CAGNNs) to enhance the performance of most GNNs on heterophily datasets by learning the neighbor effect for each node. Specifically, we first decouple the feature of each node into the discriminative feature for downstream tasks and the aggregation feature for graph convolution. Then, we propose a shared mixer module to adaptively evaluate the neighbor effect of each node to incorporate the neighbor information. The proposed framework can be regarded as a plug-in component and is compatible with most GNNs. The experimental results over nine well-known benchmark datasets indicate that our framework can significantly improve performance, especially for the heterophily graphs. The average performance gain is 9.81%, 25.81%, and 20.61% compared with GIN, GAT, and GCN, respectively. Extensive ablation studies and robustness analysis further verify the effectiveness, robustness, and interpretability of our framework. 
</p>

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

## Citation
```
@article{chen2023exploiting,
  title={Exploiting Neighbor Effect: Conv-Agnostic GNN Framework for Graphs With Heterophily},
  author={Chen, Jie and Chen, Shouzhen and Gao, Junbin and Huang, Zengfeng and Zhang, Junping and Pu, Jian},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2023},
  publisher={IEEE}
}
```