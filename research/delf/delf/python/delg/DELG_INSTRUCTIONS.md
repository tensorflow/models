## DELG instructions

[![Paper](http://img.shields.io/badge/paper-arXiv.2001.05027-B3181B.svg)](https://arxiv.org/abs/2001.05027)

These instructions can be used to reproduce the results from the
[DELG paper](https://arxiv.org/abs/2001.05027) for the Revisited Oxford/Paris
datasets.

### Install DELF library

To be able to use this code, please follow
[these instructions](../../../INSTALL_INSTRUCTIONS.md) to properly install the
DELF library.

### Download datasets

```bash
mkdir -p ~/delg/data && cd ~/delg/data

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

This is necessary to reproduce the main paper results:

```bash
# From models/research/delf/delf/python/delg
mkdir parameters && cd parameters

# DELG-GLD model.
wget http://storage.googleapis.com/delf/delg_gld_20200520.tar.gz
tar -xvzf delg_gld_20200520.tar.gz
```

### Feature extraction

We present here commands for extraction on `roxford5k`. To extract on `rparis6k`
instead, please edit the arguments accordingly (especially the
`dataset_file_path` argument).

#### Query feature extraction

For query feature extraction, the cropped query image should be used to extract
features, according to the Revisited Oxford/Paris experimental protocol. Note
that this is done in the `extract_features` script, when setting
`image_set=query`.

Query feature extraction can be run as follows:

```bash
# From models/research/delf/delf/python/delg
python3 extract_features.py \
  --delf_config_path delg_gld_config.pbtxt \
  --dataset_file_path ~/delg/data/gnd_roxford5k.mat \
  --images_dir ~/delg/data/oxford5k_images \
  --image_set query \
  --output_features_dir ~/delg/data/oxford5k_features/query
```

#### Index feature extraction

Run index feature extraction as follows:

```bash
# From models/research/delf/delf/python/delg
python3 extract_features.py \
  --delf_config_path delg_gld_config.pbtxt \
  --dataset_file_path ~/delg/data/gnd_roxford5k.mat \
  --images_dir ~/delg/data/oxford5k_images \
  --image_set index \
  --output_features_dir ~/delg/data/oxford5k_features/index
```

### Perform retrieval

To run retrieval on `roxford5k`, the following command can be used:

```bash
# From models/research/delf/delf/python/delg
python3 perform_retrieval.py \
  --dataset_file_path ~/delg/data/gnd_roxford5k.mat \
  --query_features_dir ~/delg/data/oxford5k_features/query \
  --index_features_dir ~/delg/data/oxford5k_features/index \
  --output_dir ~/delg/results/oxford5k
```

A file with named `metrics.txt` will be written to the path given in
`output_dir`, with retrieval metrics for an experiment where geometric
verification is not used. The contents should look approximately like:

```
hard
  mAP=45.11
  mP@k[ 1  5 10] [85.71 72.29 60.14]
  mR@k[ 1  5 10] [19.15 29.72 36.32]
medium
  mAP=69.71
  mP@k[ 1  5 10] [95.71 92.   86.86]
  mR@k[ 1  5 10] [10.17 25.94 33.83]
```

which are the results presented in Table 3 of the paper.

If you want to run retrieval with geometric verification, set
`use_geometric_verification` to `True`. It's much slower since (1) in this code
example the re-ranking is loading DELF local features from disk, and (2)
re-ranking needs to be performed separately for each dataset protocol, since the
junk images from each protocol should be removed when re-ranking. Here is an
example command:

```bash
# From models/research/delf/delf/python/delg
python3 perform_retrieval.py \
  --dataset_file_path ~/delg/data/gnd_roxford5k.mat \
  --query_features_dir ~/delg/data/oxford5k_features/query \
  --index_features_dir ~/delg/data/oxford5k_features/index \
  --use_geometric_verification \
  --output_dir ~/delg/results/oxford5k_with_gv
```

The `metrics.txt` should now show:

```
hard
  mAP=45.11
  mP@k[ 1  5 10] [85.71 72.29 60.14]
  mR@k[ 1  5 10] [19.15 29.72 36.32]
hard_after_gv
  mAP=53.72
  mP@k[ 1  5 10] [91.43 83.81 74.38]
  mR@k[ 1  5 10] [19.45 34.45 44.64]
medium
  mAP=69.71
  mP@k[ 1  5 10] [95.71 92.   86.86]
  mR@k[ 1  5 10] [10.17 25.94 33.83]
medium_after_gv
  mAP=75.42
  mP@k[ 1  5 10] [97.14 95.24 93.81]
  mR@k[ 1  5 10] [10.21 27.21 37.72]
```

which, again, are the results presented in Table 3 of the paper.
