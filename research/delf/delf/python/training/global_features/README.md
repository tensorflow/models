## Global features: CNN Image Retrieval


This Python toolbox implements the training and testing of the approach described in the papers:

[![Paper](http://img.shields.io/badge/paper-arXiv.2001.05027-B3181B.svg)](https://arxiv.org/abs/1711.02512)

```
"Fine-tuning CNN Image Retrieval with No Human Annotation",  
Radenović F., Tolias G., Chum O.,
TPAMI 2018 
```

[![Paper](http://img.shields.io/badge/paper-arXiv.2001.05027-B3181B.svg)](http://arxiv.org/abs/1604.02426)
```
"CNN Image Retrieval Learns from BoW: Unsupervised Fine-Tuning with Hard Examples",  
Radenović F., Tolias G., Chum O.,
ECCV 2016
```

Fine-tuned CNNs are used for global feature extraction with the goal of using
those for image retrieval. The networks are trained on the <i>SfM120k</i>
landmark images dataset.

<img src="http://cmp.felk.cvut.cz/cnnimageretrieval/img/cnnimageretrieval_network_medium.png" width=\textwidth/>

When initializing the network, one of the popular pre-trained architectures
 for classification tasks (such as ResNet or VGG) is used as the network’s
  backbone. The
fully connected layers of such architectures are discarded, resulting in a fully
convolutional backbone. Then, given an input image of the size [W × H × C],
where C is the number of channels, W and H are image width and height,
respectively; the output is a tensor X with dimensions [W' × H' × K], where
K is the number of feature maps in the last layer. Tensor X
can be considered as a set of the input image’s deep local features. For
deep convolutional features, the simple aggregation approach based on global
pooling arguably provides the best results. This method is fast, has a small
number of parameters, and a low risk of overfitting. Keeping this in mind,
we convert local features to a global descriptor vector using one of the
retrieval system’s global poolings (MAC, SPoC, or GeM). After this stage,
the feature vector is made up of the maximum activation per feature map
with dimensionality equal to K. The final output dimensionality for the most
common networks varies from 512 to 2048, making this image representation
relatively compact.

Vectors that have been pooled are subsequently L2-normalized. The obtained
 representation is then optionally passed through the fully connected
layers before being subjected to a
new L2 re-normalization. The finally produced image representation allows
comparing the resemblance of two images by simply using their inner product.


### Install DELF library

To be able to use this code, please follow
[these instructions](../../../../INSTALL_INSTRUCTIONS.md) to properly install
the DELF library.

### Usage

<details>
  <summary><b>Training</b></summary><br/>
  
  Navigate (```cd```) to the folder ```[DELF_ROOT/delf/python/training
  /global_features].```
  Example training script is located in ```DELF_ROOT/delf/python/training/global_features/train.py```.
  ```
  python3 train.py [--arch ARCH] [--batch_size N] [--data_root PATH]
          [--debug] [--directory PATH] [--epochs N] [--gpu_id ID] 
          [--image_size SIZE] [--launch_tensorboard] [--loss LOSS] 
          [--loss_margin LM] [--lr LR] [--momentum M] [multiscale SCALES] 
          [--neg_num N] [--optimizer OPTIMIZER] [--pool POOL] [--pool_size N]
          [--pretrained] [--precompute_whitening DATASET] [--resume]
          [--query_size N] [--test_datasets DATASET] [--test_freq N]
          [--test_whiten] [--training_dataset DATASET] [--update_every N]
          [--validation_type TYPE] [--weight_decay N] [--whitening]
  ```

  For detailed explanation of the options run:
  ```
  python3 train.py -helpfull
  ```
  Standard training of our models was run with the following parameters:
  ```
python3 train.py \
--directory="DESTINATION_PATH" \
--gpu_ids='0' \
--data_root="TRAINING_DATA_DIRECTORY" \
--training_dataset='retrieval-SfM-120k' \
--test_datasets='roxford5k,rparis6k' \
--arch='ResNet101' \
--pool='gem' \
--whitening=True \
--debug=True \
--loss='triplet' \
--loss_margin=0.85 \
--optimizer='adam' \
--lr=5e-7 --neg_num=3 --query_size=2000 \
--pool_size=20000 --batch_size=5 \
--image_size=1024 --epochs=100 --test_freq=5 \
--multiscale='[1, 2**(1/2), 1/2**(1/2)]'
```

  **Note**: Data and networks used for training and testing are automatically downloaded when using the example training
   script (```DELF_ROOT/delf/python/training/global_features/train.py```).

</details>

<details>
<summary><b>Training logic flow</b></summary><br/>

**Initialization phase**

1. Checking if required datasets are downloaded and automatically download them (both test and train/val) if they are 
not present in the data folder.
1. Setting up the logging and creating a logging/checkpoint directory.
1. Initialize model according to the user-provided parameters (architecture
/pooling/whitening/pretrained etc.).
1. Defining loss (contrastive/triplet) according to the user parameters.
1. Defining optimizer (Adam/SGD with learning rate/weight decay/momentum) according to the user parameters.
1. Initializing CheckpointManager and resuming from the latest checkpoint if the resume flag is set.
1. Launching Tensorboard if the flag is set.
1. Initializing training (and validation, if required) datasets.
1. Freezing BatchNorm weights update, since we we do training for one image at a time so the statistics would not be per batch, hence we choose freezing (i.e., using pretrained imagenet statistics).
1. Evaluating the network performance before training (on the test datasets).

**Training phase**

The main training loop (for the required number of epochs):
1. Finding the hard negative pairs in the dataset (using the forward pass through the model)
1. Creating the training dataset from generator which changes every epoch. Each
 element in the dataset consists of 1 x Positive image, 1 x Query image
 , N x Hard negative images (N is specified by the `num_neg` flag), an array
  specifying the Positive (-1), Query (0), Negative (1) images.
1. Performing one training step and calculating the final epoch loss.
1. If validation is required, finding hard negatives in the validation set
, which has the same structure as the training set. Performing one validation
 step and calculating the loss.
1. Evaluating on the test datasets every `test_freq` epochs.
1. Saving checkpoint (optimizer and the model weights).

</details>

## Exporting the Trained Model

Assuming the training output, the TensorFlow checkpoint, is located in the
`--directory` path. The following code exports the model:
```
python3 model/export_CNN_global_model.py \
        [--ckpt_path PATH] [--export_path PATH] [--input_scales_list LIST]
        [--multi_scale_pool_type TYPE] [--normalize_global_descriptor BOOL] 
        [arch ARCHITECTURE] [pool POOLING] [whitening BOOL]
```
*NOTE:* Path to the checkpoint must include .h5 file.

## Testing the trained model
After the trained model has been exported, it can be used to extract global
features similarly as for the DELG model. Please follow 
[these instructions](https://github.com/tensorflow/models/tree/master/research/delf/delf/python/training#testing-the-trained-model).

After training the standard training setup for 100 epochs, the
 following results are obtained on Roxford and RParis datasets under a single
 -scale evaluation:
```
>> roxford5k: mAP E: 74.88, M: 58.28, H: 30.4
>> roxford5k: mP@k[1, 5, 10] E: [89.71 84.8  79.07],
                             M: [91.43 84.67 78.24],
                             H: [68.57 53.29 43.29]

>> rparis6k: mAP E: 89.21, M: 73.69, H: 49.1
>> rparis6k: mP@k[1, 5, 10] E: [98.57 97.43 95.57],
                            M: [98.57 99.14 98.14],
                            H: [94.29 90.   87.29]
```