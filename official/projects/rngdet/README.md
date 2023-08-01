# Road Network Graph Detection by Transformer

[![Paper](https://img.shields.io/badge/Paper-TGRS.2022.3186993-00629B?logo=ieee)](https://ieeexplore.ieee.org/abstract/document/9810294)
[![Paper](https://img.shields.io/badge/Paper-LRA.2023.3264723-00629B?logo=ieee)](https://ieeexplore.ieee.org/abstract/document/10093124)

## Environment setup
The code can be run on multiple GPUs or TPUs with different distribution
strategies. See the TensorFlow distributed training
[guide](https://www.tensorflow.org/guide/distributed_training) for an overview
of `tf.distribute`.

## Data preparation
To download the dataset and generate labels, try the following command:

```
cd data
./prepare_dataset.bash
```

To generate training samples, try the following command:

```
python create_cityscale_tf_record.py \
    --dataroot ./dataset/ \
    --roi_size 128 \
    --image_size 2048 \
    --edge_move_ahead_length 30 \
    --num_queries 10 \
    --noise 8 \
    --max_num_frame 10000 \
    --num_shards 32
```