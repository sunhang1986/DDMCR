Dual Attention Mechanism and Contrastive Learning Constraint for Point Cloud Classification
==
Created by [Yuanyue Zhang](https://github.com/yy-zhang832), [Hang Sun](https://github.com/sunhang1986), Shuifa Sun from Department of Computer and Information, China Three Gorges University.

Introduction
--
Our paper proposes a dual attention mechanism and contrastive learning constraint point cloud classification network(DACR-Net). It can benefit from channel and space to strengthen the interaction between local features respectively and improve the capability of feature extraction of the network.

Prerequisites
--
+ Tensorflow 1.3.0
+ Python 3.6
+ CUDA 8.0
+ Ubuntu 18.04

Test
--
[Download](https://1drv.ms/u/s!ApbTjxa06z9CgQfKl99yUDHL_wHs) *ModelNet-40* dataset first. Point clouds are sampled from meshes with 10K points (XYZ + normals) per shape and provided by PointNet++.

Test the model on *ModelNet-40*:

` python test.py` 

