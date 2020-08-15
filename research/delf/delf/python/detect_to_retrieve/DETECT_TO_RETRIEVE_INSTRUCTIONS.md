## Detect-to-Retrieve instructions

[![Paper](http://img.shields.io/badge/paper-arXiv.1812.01584-B3181B.svg)](https://arxiv.org/abs/1812.01584)

These instructions can be used to reproduce the results from the
[Detect-to-Retrieve paper](https://arxiv.org/abs/1812.01584) for the Revisited
Oxford/Paris datasets.

### Install DELF library

To be able to use this code, please follow
[these instructions](../../../INSTALL_INSTRUCTIONS.md) to properly install the
DELF library.

### Download datasets

```bash
mkdir -p ~/detect_to_retrieve/data && cd ~/detect_to_retrieve/data

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

### Download models

These are necessary to reproduce the main paper results:

```bash
# From models/research/delf/delf/python/detect_to_retrieve
mkdir parameters && cd parameters

# DELF-GLD model.
wget http://storage.googleapis.com/delf/delf_gld_20190411.tar.gz
tar -xvzf delf_gld_20190411.tar.gz

# Faster-RCNN detector model.
wget http://storage.googleapis.com/delf/d2r_frcnn_20190411.tar.gz
tar -xvzf d2r_frcnn_20190411.tar.gz

# Codebooks.
# Note: you should use codebook trained on rparis6k for roxford5k retrieval
# experiments, and vice-versa.
wget http://storage.googleapis.com/delf/rparis6k_codebook_65536.tar.gz
mkdir rparis6k_codebook_65536
tar -xvzf rparis6k_codebook_65536.tar.gz -C rparis6k_codebook_65536/
wget http://storage.googleapis.com/delf/roxford5k_codebook_65536.tar.gz
mkdir roxford5k_codebook_65536
tar -xvzf roxford5k_codebook_65536.tar.gz -C roxford5k_codebook_65536/
```

We also make available other models/parameters that can be used to reproduce
more results from the paper:

-   [MobileNet-SSD trained detector](http://storage.googleapis.com/delf/d2r_mnetssd_20190411.tar.gz).
-   Codebooks with 1024 centroids:
    [rparis6k](http://storage.googleapis.com/delf/rparis6k_codebook_1024.tar.gz),
    [roxford5k](http://storage.googleapis.com/delf/roxford5k_codebook_1024.tar.gz).

### Feature extraction

We present here commands for extraction on `roxford5k`. To extract on `rparis6k`
instead, please edit the arguments accordingly (especially the
`dataset_file_path` argument).

#### Query feature extraction

For query feature extraction, the cropped query image should be used to extract
features, according to the Revisited Oxford/Paris experimental protocol. Note
that this is done in the `extract_query_features` script.

Query feature extraction can be run as follows:

```bash
# From models/research/delf/delf/python/detect_to_retrieve
python3 extract_query_features.py \
  --delf_config_path delf_gld_config.pbtxt \
  --dataset_file_path ~/detect_to_retrieve/data/gnd_roxford5k.mat \
  --images_dir ~/detect_to_retrieve/data/oxford5k_images \
  --output_features_dir ~/detect_to_retrieve/data/oxford5k_features/query
```

#### Index feature extraction and box detection

Index feature extraction / box detection can be run as follows:

```bash
# From models/research/delf/delf/python/detect_to_retrieve
python3 extract_index_boxes_and_features.py \
  --delf_config_path delf_gld_config.pbtxt \
  --detector_model_dir parameters/d2r_frcnn_20190411 \
  --detector_thresh 0.1 \
  --dataset_file_path ~/detect_to_retrieve/data/gnd_roxford5k.mat \
  --images_dir ~/detect_to_retrieve/data/oxford5k_images \
  --output_boxes_dir ~/detect_to_retrieve/data/oxford5k_boxes/index \
  --output_features_dir ~/detect_to_retrieve/data/oxford5k_features/index_0.1 \
  --output_index_mapping ~/detect_to_retrieve/data/oxford5k_features/index_mapping_0.1.csv
```

### R-ASMK* aggregation extraction

We present here commands for aggregation extraction on `roxford5k`. To extract
on `rparis6k` instead, please edit the arguments accordingly. In particular,
note that feature aggregation on `roxford5k` should use a codebook trained on
`rparis6k`, and vice-versa (this can be edited in the
`query_aggregation_config.pbtxt` and `index_aggregation_config.pbtxt` files.

#### Query

Run query feature aggregation as follows:

```bash
# From models/research/delf/delf/python/detect_to_retrieve
python3 extract_aggregation.py \
  --use_query_images True \
  --aggregation_config_path query_aggregation_config.pbtxt \
  --dataset_file_path ~/detect_to_retrieve/data/gnd_roxford5k.mat \
  --features_dir ~/detect_to_retrieve/data/oxford5k_features/query \
  --output_aggregation_dir ~/detect_to_retrieve/data/oxford5k_aggregation/query
```

#### Index

Run index feature aggregation as follows:

```bash
# From models/research/delf/delf/python/detect_to_retrieve
python3 extract_aggregation.py \
  --aggregation_config_path index_aggregation_config.pbtxt \
  --dataset_file_path ~/detect_to_retrieve/data/gnd_roxford5k.mat \
  --features_dir ~/detect_to_retrieve/data/oxford5k_features/index_0.1 \
  --index_mapping_path ~/detect_to_retrieve/data/oxford5k_features/index_mapping_0.1.csv \
  --output_aggregation_dir ~/detect_to_retrieve/data/oxford5k_aggregation/index_0.1
```

### Perform retrieval

Currently, we support retrieval via brute-force comparison of aggregated
features.

To run retrieval on `roxford5k`, the following command can be used:

```bash
# From models/research/delf/delf/python/detect_to_retrieve
python3 perform_retrieval.py \
  --index_aggregation_config_path index_aggregation_config.pbtxt \
  --query_aggregation_config_path query_aggregation_config.pbtxt \
  --dataset_file_path ~/detect_to_retrieve/data/gnd_roxford5k.mat \
  --index_aggregation_dir ~/detect_to_retrieve/data/oxford5k_aggregation/index_0.1 \
  --query_aggregation_dir ~/detect_to_retrieve/data/oxford5k_aggregation/query \
  --output_dir ~/detect_to_retrieve/results/oxford5k
```

A file with named `metrics.txt` will be written to the path given in
`output_dir`, with retrieval metrics for an experiment where geometric
verification is not used. The contents should look approximately like:

```
hard
mAP=47.61
mP@k[ 1  5 10] [84.29 73.71 64.43]
mR@k[ 1  5 10] [18.84 29.44 36.82]
medium
mAP=73.3
mP@k[ 1  5 10] [97.14 94.57 90.14]
mR@k[ 1  5 10] [10.14 26.2  34.75]
```

which are the results presented in Table 2 of the paper (with small numerical
precision differences).

If you want to run retrieval with geometric verification, set
`use_geometric_verification` to `True` and the arguments
`index_features_dir`/`query_features_dir`. It's much slower since (1) in this
code example the re-ranking is loading DELF local features from disk, and (2)
re-ranking needs to be performed separately for each dataset protocol, since the
junk images from each protocol should be removed when re-ranking. Here is an
example command:

```bash
# From models/research/delf/delf/python/detect_to_retrieve
python3 perform_retrieval.py \
  --index_aggregation_config_path index_aggregation_config.pbtxt \
  --query_aggregation_config_path query_aggregation_config.pbtxt \
  --dataset_file_path ~/detect_to_retrieve/data/gnd_roxford5k.mat \
  --index_aggregation_dir ~/detect_to_retrieve/data/oxford5k_aggregation/index_0.1 \
  --query_aggregation_dir ~/detect_to_retrieve/data/oxford5k_aggregation/query \
  --use_geometric_verification True \
  --index_features_dir ~/detect_to_retrieve/data/oxford5k_features/index_0.1 \
  --query_features_dir ~/detect_to_retrieve/data/oxford5k_features/query \
  --output_dir ~/detect_to_retrieve/results/oxford5k_with_gv
```

### Clustering

In the code example above, we used a pre-trained DELF codebook. We also provide
code for re-training the codebook if desired.

Note that for the time being this can only run on CPU, since the main ops in
K-means are not registered for GPU usage in Tensorflow.

```bash
# From models/research/delf/delf/python/detect_to_retrieve
python3 cluster_delf_features.py \
  --dataset_file_path ~/detect_to_retrieve/data/gnd_rparis6k.mat \
  --features_dir ~/detect_to_retrieve/data/paris6k_features/index_0.1 \
  --num_clusters 1024 \
  --num_iterations 50 \
  --output_cluster_dir ~/detect_to_retrieve/data/paris6k_clusters_1024
```

### Next steps

To make retrieval more scalable and handle larger datasets more smoothly, we are
considering to provide code for inverted index building and retrieval. Please
reach out if you would like to help doing that -- feel free submit a pull
request.
