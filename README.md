## Purple Wizard

# BPR

### Introduction

This is a bachelor project repository for group Purple Wizard. The project aims to create a pipeline for dimensionality reduction while preserving the semantic relationship between them. As such, the reduced vectors should be able to fully restore and should be suitable to use in downstream tasks.

This imposes two issues:

* Increase lookup time when performing semantic search
* increase storage requirements

### How to

Copy unzipped version of tiny-imagenet-200 into /dataset

``python3 main.py``

### Dataset
The project uses the WorldStrat dataset which consists of 39 images. The dataset consists of ------ images of which ----- is used as training set and ----- as test set. https://www.kaggle.com/datasets/jucor1/worldstrat

### References

Hashing with binary Autoencoders: https://arxiv.org/pdf/1501.00756.pdf

Barlow Twins: https://github.com/sayakpaul/Barlow-Twins-TF


### Setup

Create virtual environment:

