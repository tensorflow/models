## GLDv2 code/models

[![Paper](http://img.shields.io/badge/paper-arXiv.2004.01804-B3181B.svg)](https://arxiv.org/abs/2004.01804)

These instructions can be used to reproduce results from the
[GLDv2 paper](https://arxiv.org/abs/2004.01804). We present here results on the
Revisited Oxford/Paris datasets since they are smaller and quicker to
reproduce -- but note that a very similar procedure can be used to obtain
results on the GLDv2 retrieval or recognition datasets.

Note that this directory also contains code to compute GLDv2 metrics: see
`compute_retrieval_metrics.py`, `compute_recognition_metrics.py` and associated
file reading / metric computation modules.

For more details on the dataset, please refer to its
[website](https://github.com/cvdfoundation/google-landmark).

### Install DELF library

To be able to use this code, please follow
[these instructions](../../../INSTALL_INSTRUCTIONS.md) to properly install the
DELF library.

### Download Revisited Oxford/Paris datasets

```bash
mkdir -p ~/revisitop/data && cd ~/revisitop/data

# Oxford dataset.
wget http://www.robots.ox.ac.uk/~vgg/data/oxbuildings/oxbuild_images.tgz
mkdir oxford5k_images
tar -xvzf oxbuild_images.tgz -C oxford5k_images/

# Paris dataset. Download and move all images to same directory.
wget http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_1.tgz
wget http://www.robots.ox.ac.uk/~vgg/data/parisbuildings/paris_2.tgz
mkdir paris6k_images_tmp
tar -xvzf paris_1.tgz -C paris6k_images_tmp/
tar -xvzf paris_2.tgz -C paris6k_images_tmp/
mkdir paris6k_images
mv paris6k_images_tmp/paris/*/*.jpg paris6k_images/

# Revisited annotations.
wget http://cmp.felk.cvut.cz/revisitop/data/datasets/roxford5k/gnd_roxford5k.mat
wget http://cmp.felk.cvut.cz/revisitop/data/datasets/rparis6k/gnd_rparis6k.mat
```

### Download model

```bash
# From models/research/delf/delf/python/google_landmarks_dataset
mkdir parameters && cd parameters

# RN101-ArcFace model trained on GLDv2-clean.
wget https://storage.googleapis.com/delf/rn101_af_gldv2clean_20200814.tar.gz
tar -xvzf rn101_af_gldv2clean_20200814.tar.gz
```

### Feature extraction

We present here commands for extraction on `roxford5k`. To extract on `rparis6k`
instead, please edit the arguments accordingly (especially the
`dataset_file_path` argument).

#### Query feature extraction

In the Revisited Oxford/Paris experimental protocol, query images must be the
cropped before feature extraction (this is done in the `extract_features`
script, when setting `image_set=query`). Note that this is specific to these
datasets, and not required for the GLDv2 retrieval/recognition datasets.

Run query feature extraction as follows:

```bash
# From models/research/delf/delf/python/google_landmarks_dataset
python3 ../delg/extract_features.py \
  --delf_config_path rn101_af_gldv2clean_config.pbtxt \
  --dataset_file_path ~/revisitop/data/gnd_roxford5k.mat \
  --images_dir ~/revisitop/data/oxford5k_images \
  --image_set query \
  --output_features_dir ~/revisitop/data/oxford5k_features/query
```

#### Index feature extraction

Run index feature extraction as follows:

```bash
# From models/research/delf/delf/python/google_landmarks_dataset
python3 ../delg/extract_features.py \
  --delf_config_path rn101_af_gldv2clean_config.pbtxt \
  --dataset_file_path ~/revisitop/data/gnd_roxford5k.mat \
  --images_dir ~/revisitop/data/oxford5k_images \
  --image_set index \
  --output_features_dir ~/revisitop/data/oxford5k_features/index
```

### Perform retrieval

To run retrieval on `roxford5k`, the following command can be used:

```bash
# From models/research/delf/delf/python/google_landmarks_dataset
python3 ../delg/perform_retrieval.py \
  --dataset_file_path ~/revisitop/data/gnd_roxford5k.mat \
  --query_features_dir ~/revisitop/data/oxford5k_features/query \
  --index_features_dir ~/revisitop/data/oxford5k_features/index \
  --output_dir ~/revisitop/results/oxford5k
```

A file with named `metrics.txt` will be written to the path given in
`output_dir`. The contents should look approximately like:

```
hard
  mAP=55.54
  mP@k[ 1  5 10] [88.57 80.86 70.14]
  mR@k[ 1  5 10] [19.46 33.65 42.44]
medium
  mAP=76.23
  mP@k[ 1  5 10] [95.71 92.86 90.43]
  mR@k[ 1  5 10] [10.17 25.96 35.29]
```
