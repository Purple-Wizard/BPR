## Purple Wizard

# BPR

### Introduction

This is a bachelor project repository for group Purple Wizard. The project aims to create a pipeline for dimensionality reduction while preserving the semantic relationship between them. As such, the reduced vectors should be able to fully restore and should be suitable to use in downstream tasks.

This imposes two issues:

* Increase lookup time when performing semantic search
* increase storage requirements

### Dataset
The project uses the WorldStrat dataset which consists of 3928 images. The original images were split grid-style to get 64 pcs of 128x128 images from a single 1054x1054 image, this way there is appr. 176000 images in the dataset that was used for training, validation and test. https://www.kaggle.com/datasets/jucor1/worldstrat

### References

Hashing with binary Autoencoders: https://arxiv.org/pdf/1501.00756.pdf

Barlow Twins: https://github.com/sayakpaul/Barlow-Twins-TF

### Setup

For image compression and decompression functionality.

##### Prerequisites

- Docker | >=Python 3.9
- Folder with .png images (more images = more resource needed, try with 5 pcs)
- ONLY for training and Barlow Twins validation: dataset (into /dataset)

#### Docker

1. Build container:

``docker build -t image-compression .``

2. Compress:

```
docker run -d -v /Path/to/project/folder:/app image-compression compress --data_to_compress test/gmaps_test --name_images sample --encoder_model models/encoder.h5 --save_locally True
```

3. Decompress:

```
docker run -d -v /Path/to/project/folder:/app image-compression decompress --decoder_model models/decoder.h5 --path_to_compressed test/compressed/test.npy --show_images 0 --save_local True --images_folder_name sample
```

#### Python/pip

1. Open the terminal and install virtual environment:

``pip3 install virtualenv``

2. Create virtual environment:

```
cd path\to\your\project
virtualenv environment_name
```

3. Activate virtual environment:

``.\env_name\Scripts\activate`` for Windows
``source env_name/bin/activate`` for Linux

``deactivate`` Stopping the virtual environment

4. Install the dependecies:

``setup.sh``

5. (Optional) Train the model:

``train.sh``

6. (Optional) Test model output:

``pipeline.py``
``barlow_twins.py``

7. Compress and decompress files:

```
main.sh compress --data_to_compress test/gmaps_test --name_images sample --encoder_model models/encoder.h5 --save_locally True
```

```
main.sh decompress --decoder_model models/decoder.h5 --path_to_compressed test/compressed/test.npy --show_images 0 --save_local True --images_folder_name sample
```

8. Compress files (parameters: data_to_compress, name_images, encoder_model, save_local), takes a folder of .png files as input:

```
compress.sh --data_to_compress test/gmaps_test --name_images sample --encoder_model models/encoder.h5 --save_locally True
```

9. Decompress files (parameters: decoder_model, path_to_compressed, show_images, save_local, images_folder_name), takes an .npy file as input:

```
decompress.sh --decoder_model models/decoder.h5 --path_to_compressed test/compressed/test.npy --show_images 0 --save_local True --images_folder_name sample
```

