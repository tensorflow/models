# Script to download models to initialize the RGB and D models for training.We
# use ResNet-v2-50 for both modalities.

mkdir -p data/init_models
cd data/init_models

# RGB Models are initialized by pre-training on ImageNet.
mkdir -p resnet_v2_50
RGB_URL="http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz"
wget $RGB_URL
tar -xf resnet_v2_50_2017_04_14.tar.gz -C resnet_v2_50

# Depth models are initialized by distilling the RGB model to D images using
# Cross-Modal Distillation (https://arxiv.org/abs/1507.00448).
mkdir -p distill_rgb_to_d_resnet_v2_50
D_URL="http://download.tensorflow.org/models/cognitive_mapping_and_planning/2017_04_16/distill_rgb_to_d_resnet_v2_50.tar"
wget $D_URL
tar -xf distill_rgb_to_d_resnet_v2_50.tar -C distill_rgb_to_d_resnet_v2_50
