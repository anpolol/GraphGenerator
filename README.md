# GraphGenerator 
This is the generator of graphs based on [BTER model](https://arxiv.org/pdf/1302.6636.pdf). The overall scheme of the generator is presented on the picture:

<p align="center">
  <img src="https://github.com/anpolol/GraphGenerator/blob/main/docs/algo.png?raw=true" width="300px"> 
</p>

* First, degrees of nodes are generated and divided on in-degree and out-degree values to keep the desired assortativity (in MyModel.py file in making_degree_dist func)
* These degrees are separeted for blocks to keep the desired number of labels (in MyModel.py file in making_clusters func for the same sizes of each label and making_clusters_with_sizes for given sizes of each label)  
* Then, in each group of nodes with the same label, the edges are generated according to BTER model (which is in BTER.py file) on in-degree nodes (in MyModel.py file bter_model_edges func)
* In the end, edges on out-degrees are generated for all nodes at ones

As we are aim at graphs with given four graph characteristics (label assortativity, feature assortativity, clustering coefficient, average length of shortest paths, average degree), so we tune all input hyperparameters of generator so that the generated graph is corresponds to the specified characteristics (in BTER_tuning.ipynb file)

## Citing
Please cite [our paper](http://www.mlgworkshop.org/2022/papers/MLG22_paper_5068.pdf) (and the respective papers of the methods used) if you use this code in your own work:
```
@inproceedings{mlg2022_5068,
title={Attributed Labeled BTER-Based Generative Model for Benchmarking of Graph Neural Networks},
author={Polina Andreeva, Egor Shikov and Claudie Bocheninа},
booktitle={Proceedings of the 17th International Workshop on Mining and Learning with Graphs (MLG)},
year={2022}
}
```
