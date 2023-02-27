=============================
GraphGenerator 
=============================

.. start-badges
.. list-table::
   :stub-columns: 1

   * - package
     - | |pypi|
   * - tests
     - | |build| |coverage|
   * - docs
     - |docs|
   * - license
     - | |license|
   * - stats
     - | |downloads_stats|
   * - support
     - | |tg|
   * - languages
     - | |eng| |rus|
     - 
.. end-badges

**GraphGenerator** is aт open-source tool for generating Graphs structure. 
It based on BTER(https://arxiv.org/pdf/1302.6636.pdf) model and provides fair and representative graphs.
This graphs can be useful in Deep Graph Learning problems, e.g. you can use it in GNN benchmarking.

Core features
-------------

To generate graph, **GraphGenerator** is using 5 characteristics: 
* _Label assortativity_
* _Feature assortativity_ 
* _Clustering coefficient_
* _Average length of the shortest paths_
* _Average degree_

Examples
--------
<p align="center">
  <img src="https://github.com/anpolol/GraphGenerator/blob/main/docs/algo.png?raw=true" width="300px"> 
</p>

The overall scheme of the generator is presented on the picture:

* Degrees of nodes are generated and divided on `in-degree` and `out-degree` values to keep the desired assortativity - `making_degree_dist()`(TODO: link in code)
* These values are separated for blocks to keep the desired number of labels - `making_clusters()`(TODO: link in code) or `making_clusters_with_sizes()`(TODO: link in code)  
* For each group of nodes with the same label, the edges are generated according to `BTER` model on `in-degree nodes` - `bter_model_edges()`
* At the end, `edges on out-degrees` are generated for all nodes at ones

The usage is presented in `BTER_tuning.ipynb`: As we are aim at graphs with given four graph characteristics,
so we tune all input hyperparameters of generator so that the generated graph corresponds
to the specified characteristics

License
-------
Lorem Ipsum

Contacts
--------
- email@itmo.ru write us to email and we will answer at your questions

Reference Paper
--------
Polina Andreeva, Egor Shikov and Claudie Bocheninа 
["Attributed Labeled BTER-Based Generative Model for Benchmarking of Graph Neural Networks"](http://www.mlgworkshop.org/2022/papers/MLG22_paper_5068.pdf)
Proceedings of the 17th International Workshop on Mining and Learning with Graphs (MLG) 2022:

```
@inproceedings{"mlg2022_5068",
title={Attributed Labeled BTER-Based Generative Model for Benchmarking of Graph Neural Networks},
author={Polina Andreeva, Egor Shikov and Claudie Bocheninа},
booktitle={Proceedings of the 17th International Workshop on Mining and Learning with Graphs (MLG)},
year={2022}
}
```
