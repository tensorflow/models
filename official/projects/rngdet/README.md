
# Road Network Graph Detection by Transformer

[![RNGDet](https://img.shields.io/badge/RNGDet-arXiv.2202.07824-B3181B?)](https://arxiv.org/abs/2202.07824)
[![RNGDet++](https://img.shields.io/badge/RNGDet++-arXiv.2209.10150-B3181B?)](https://arxiv.org/abs/2209.10150)

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
## Training 
To edit training options of RNGDet, you can edit following commands in do_train.sh :

```
CUDA_VISIBLE_DEVICES=4 python3 train.py \
  --mode=train \
  --experiment=rngdet_cityscale  \
  --model_dir=./CKPT_DIR_NAME \
  --config_file=./configs/experiments/cityscale_rngdet_r50_gpu.yaml \
```

To start training, try the following command : 
```
sh do_train.sh 
```

## Evaluation 
To evaluate one image with internal step visualization,  

```
python run_rngdet_basic.py -ckpt ./CKPT_DIR_NAME
```
 
