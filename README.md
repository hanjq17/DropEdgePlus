# Structure-Aware DropEdge Towards Deep Graph Convolutional Networks (TNNLS)

Jiaqi Han, Wenbing Huang, Yu Rong, Tingyang Xu, Fuchun Sun, Junzhou Huang

In IEEE Transactions on Neural Networks and Learning Systems, 2023.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/hanjq17/DropEdgePlus/blob/main/LICENSE)

[**[Paper]**](https://doi.org/10.1109/TNNLS.2023.3288484)

In Structure-Aware DropEdge, we enhance graph edge dropping technique with two structure-aware samplers, the layer-dependent sampler and feature-dependent sampler, to further relieve the over-smoothing issue in deep graph networks.

<!-- ![Overview](assets/overview.png "Overview") -->



## Dependencies


Please check out the Python environment depicted in `requirements.txt`.



## Data

The semi-supervised setting strictly follows [GCN](https://github.com/tkipf/gcn), and the full-supervised setting follows [DropEdge](https://github.com/DropEdge/DropEdge). The co-author and co-purchase datasets can be downloaded from https://github.com/shchur/gnn-benchmark. 



## Running the Experiments


The code has been tested in the above-mentioned environment with `Python=3.6.2`. We recommend using conda.

```bash
conda create -n xxx python=3.6.2
conda activate xxx
pip install -r requirements.txt
```

To reproduce our results, just run the scripts in the `scripts` folder. For example,

```bash
sh scripts/semi/citeseer_appnp.sh
```

## Citation

If you find our work helpful, please cite as:

```
@ARTICLE{10195874,
  author={Han, Jiaqi and Huang, Wenbing and Rong, Yu and Xu, Tingyang and Sun, Fuchun and Huang, Junzhou},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Structure-Aware DropEdge Toward Deep Graph Convolutional Networks}, 
  year={2023},
  volume={},
  number={},
  pages={1-13},
  doi={10.1109/TNNLS.2023.3288484}}

```


## Contact

If you have any questions, feel free to reach us at:

Jiaqi Han: alexhan99max@gmail.com
