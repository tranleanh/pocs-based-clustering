# POCS-based Clustering Algorithm 

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://tranleanh.github.io/assets/pdf/IWIS_2022.pdf)
[![Paper](https://img.shields.io/badge/Conference-IEEE-blue)](https://ieeexplore.ieee.org/document/9920762)
[![Blog](https://img.shields.io/badge/Blog-Medium-blue)](https://towardsdatascience.com/pocs-based-clustering-algorithm-explained-2f7d25183eff)
[![Blog](https://img.shields.io/badge/Package-PyPI-white)](https://pypi.org/project/pocs-based-clustering/)


Official implementation of the Projection Onto Convex Set (POCS)-based clustering algorithm.

## Introduction

<!--- - Paper: [POCS-based Clustering Algorithm](https://ieeexplore.ieee.org/abstract/document/9920762) (in Proceedings of The International Workshop on Intelligent Systems 2022 (IWIS 2022), Ulsan, South Korea, August 17-19, 2022) --->

 - Main authors: [Le-Anh Tran](https://scholar.google.com/citations?user=WzcUE5YAAAAJ&hl=en), [Dong-Chul Park](https://scholar.google.com/citations?user=VZUH4sUAAAAJ&hl=en)
 <!--- - Blog: [POCS-based Clustering Algorithm Explained (Medium)](https://tranlevision.medium.com/pocs-based-clustering-algorithm-explained-2f7d25183eff) --->
 - Algorithm:
     <p align="center">
     <img src="docs/pocs_based_clustering_pseudocode.png" width="750">
     </p>

## Usage

### Package

We release a package of this project on [PyPI](https://pypi.org/project/pocs-based-clustering/):

```
pip install pocs-based-clustering
```
    
### Demo via Installed Package


```
test_pocs_clustering_package.ipynb
```

### Demo via Local Script

```
python test_pocs_clustering_local.py
```
    
## Results

Results on [Clustering basic benchmark](https://cs.joensuu.fi/sipu/datasets/)

<p align="center">
<img src="docs/results_synthetic_datasets.jpg" width="1000">
</p>


## Citation

Please cite our works if you utilize any data from this work for your study.

```bibtex
@inproceedings{tran2022pocs,
  title={POCS-based Clustering Algorithm},
  author={Tran, Le-Anh and Deberneh, Henock M and Do, Truong-Dong and Nguyen, Thanh-Dat and Le, My-Ha and Park, Dong-Chul},
  booktitle={2022 International Workshop on Intelligent Systems (IWIS)},
  pages={1--6},
  year={2022},
  organization={IEEE}
}

@article{tran2024cluster,
  title={Cluster Analysis via Projection onto Convex Sets},
  author={Tran, Le-Anh and Kwon, Daehyun and Deberneh, Henock Mamo and Park, Dong-Chul},
  journal={Intelligent Data Analysis},
  year={2024},
  publisher={IOS Press}
}
```

Have fun!

LA Tran

<!--- Created: 04.2022 --->
