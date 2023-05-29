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
The project uses the ImageNet dataset which consists of --*-- images. The dataset consists of ------ images of which ----- is used as training set and ----- as test set.

### Usages

The following libraries has been used in the project:

* NumPy
* Pandas
* Scikit-learn
* TensorFlow
* Keras

### References

Hashing with binary Autoencoders: https://arxiv.org/pdf/1501.00756.pdf

Airflow docs: https://airflow.apache.org/docs/apache-airflow/stable/

Nifi: https://nifi.apache.org/

Ceph docs: https://docs.ceph.com/en/quincy/


### Setup

Create virtual environment:

