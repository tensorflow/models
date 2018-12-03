# struct2depth

This a method for unsupervised learning of depth and egomotion from monocular video, achieving new state-of-the-art results on both tasks by explicitly modeling 3D object motion, performing on-line refinement and improving quality for moving objects by novel loss formulations. It will appear in the following paper: 

**V. Casser, S. Pirk, R. Mahjourian, A. Angelova, Depth Prediction Without the Sensors: Leveraging Structure for Unsupervised Learning from Monocular Videos, AAAI Conference on Artificial Intelligence, 2019**
https://arxiv.org/pdf/1811.06152.pdf

This code is implemented and supported by Vincent Casser (git username: VincentCa) and Anelia Angelova (git username: AneliaAngelova). Please contact anelia@google.com for questions. 

Project website: https://sites.google.com/view/struct2depth.

## Quick start: Running training

Before running training, run gen_data_* script for the respective dataset in order to generate the data in the appropriate format for KITTI or Cityscapes. It is assumed that motion masks are already generated and stored as images.
Models are trained from an Imagenet pretrained model.

```shell

ckpt_dir="your/checkpoint/folder"
data_dir="KITTI_SEQ2_LR/" # Set for KITTI
data_dir="CITYSCAPES_SEQ2_LR/" # Set for Cityscapes
imagenet_ckpt="resnet_pretrained/model.ckpt"

python train.py \
  --logtostderr \
  --checkpoint_dir $ckpt_dir \
  --data_dir $data_dir \
  --architecture resnet \
  --imagenet_ckpt $imagenet_ckpt \
  --imagenet_norm true \
  --joint_encoder false
```



## Running depth/egomotion inference on an image folder

KITTI is trained on the raw image data (resized to 416 x 128), but inputs are standardized before feeding them, and Cityscapes images are cropped using the following cropping parameters: (192, 1856, 256, 768). If using a different crop, it is likely that additional training is necessary. Therefore, please follow the inference example shown below when using one of the models. The right choice might depend on a variety of factors. For example, if a checkpoint should be used for odometry, be aware that for improved odometry on motion models, using segmentation masks could be advantageous (setting *use_masks=true* for inference). On the other hand, all models can be used for single-frame depth estimation without any additional information.


```shell

input_dir="your/image/folder"
output_dir="your/output/folder"
model_checkpoint="your/model/checkpoint"

python inference.py \
    --logtostderr \
    --file_extension png \
    --depth \
    --egomotion true \
    --input_dir $input_dir \
    --output_dir $output_dir \
    --model_ckpt $model_checkpoint
```

Note that the egomotion prediction expects the files in the input directory to be a consecutive sequence, and that sorting the filenames alphabetically is putting them in the right order.

One can also run inference on KITTI by providing

```shell
--input_list_file ~/kitti-raw-uncompressed/test_files_eigen.txt
```

and on Cityscapes by passing

```shell
--input_list_file CITYSCAPES_FULL/test_files_cityscapes.txt
```

instead of *input_dir*.
Alternatively inference can also be ran on pre-processed images.



## Running on-line refinement

On-line refinement is executed on top of an existing inference folder, so make sure to run regular inference first. Then you can run the on-line fusion procedure as follows:

```shell

prediction_dir="some/prediction/dir"
model_ckpt="checkpoints/checkpoints_baseline/model-199160"
handle_motion="false"
size_constraint_weight="0" # This must be zero when not handling motion.

# If running on KITTI, set as follows:
data_dir="KITTI_SEQ2_LR_EIGEN/"
triplet_list_file="$data_dir/test_files_eigen_triplets.txt"
triplet_list_file_remains="$data_dir/test_files_eigen_triplets_remains.txt"
ft_name="kitti"

# If running on Cityscapes, set as follows:
data_dir="CITYSCAPES_SEQ2_LR_TEST/" # Set for Cityscapes
triplet_list_file="/CITYSCAPES_SEQ2_LR_TEST/test_files_cityscapes_triplets.txt"
triplet_list_file_remains="CITYSCAPES_SEQ2_LR_TEST/test_files_cityscapes_triplets_remains.txt"
ft_name="cityscapes"

python optimize.py \
  --logtostderr \
  --output_dir $prediction_dir \
  --data_dir $data_dir \
  --triplet_list_file $triplet_list_file \
  --triplet_list_file_remains $triplet_list_file_remains \
  --ft_name $ft_name \
  --model_ckpt $model_ckpt \
  --file_extension png \
  --handle_motion $handle_motion \
  --size_constraint_weight $size_constraint_weight
```



## Running evaluation

```shell

prediction_dir="some/prediction/dir"

# Use these settings for KITTI:
eval_list_file="KITTI_FULL/kitti-raw-uncompressed/test_files_eigen.txt"
eval_crop="garg"
eval_mode="kitti"

# Use these settings for Cityscapes:
eval_list_file="CITYSCAPES_FULL/test_files_cityscapes.txt"
eval_crop="none"
eval_mode="cityscapes"

python evaluate.py \
  --logtostderr \
  --prediction_dir $prediction_dir \
  --eval_list_file $eval_list_file \
  --eval_crop $eval_crop \
  --eval_mode $eval_mode
```



## Credits

This code is implemented and supported by Vincent Casser and Anelia Angelova and can be found at
https://sites.google.com/view/struct2depth.
The core implementation is derived from [https://github.com/tensorflow/models/tree/master/research/vid2depth)](https://github.com/tensorflow/models/tree/master/research/vid2depth)
by [Reza Mahjourian](rezama@google.com), which in turn is based on [SfMLearner
(https://github.com/tinghuiz/SfMLearner)](https://github.com/tinghuiz/SfMLearner)
by [Tinghui Zhou](https://github.com/tinghuiz).
