<font size=4><b>Train Wide-ResNet, Shake-Shake and ShakeDrop models on CIFAR-10
and CIFAR-100 dataset with AutoAugment.</b></font>

The CIFAR-10/CIFAR-100 data can be downloaded from:
https://www.cs.toronto.edu/~kriz/cifar.html.

The code replicates the results from Tables 1 and 2 on CIFAR-10/100 with the
following models: Wide-ResNet-28-10, Shake-Shake (26 2x32d), Shake-Shake (26
2x96d) and PyramidNet+ShakeDrop.

<b>Related papers:</b>

AutoAugment: Learning Augmentation Policies from Data

https://arxiv.org/abs/1805.09501

Wide Residual Networks

https://arxiv.org/abs/1605.07146

Shake-Shake regularization

https://arxiv.org/abs/1705.07485

ShakeDrop regularization

https://arxiv.org/abs/1802.02375

<b>Settings:</b>

CIFAR-10 Model         | Learning Rate | Weight Decay | Num. Epochs | Batch Size
---------------------- | ------------- | ------------ | ----------- | ----------
Wide-ResNet-28-10      | 0.1           | 5e-4         | 200         | 128
Shake-Shake (26 2x32d) | 0.01          | 1e-3         | 1800        | 128
Shake-Shake (26 2x96d) | 0.01          | 1e-3         | 1800        | 128
PyramidNet + ShakeDrop | 0.05          | 5e-5         | 1800        | 64

<b>Prerequisite:</b>

1.  Install TensorFlow.

2.  Download CIFAR-10/CIFAR-100 dataset.

```shell
curl -o cifar-10-binary.tar.gz https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
curl -o cifar-100-binary.tar.gz https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz
```

<b>How to run:</b>

```shell
# cd to the your workspace.
# Specify the directory where dataset is located using the data_path flag.
# Note: User can split samples from training set into the eval set by changing train_size and validation_size.

# For example, to train the Wide-ResNet-28-10 model on a GPU.
python train_cifar.py --model_name=wrn \
                      --checkpoint_dir=/tmp/training \
                      --data_path=/tmp/data \
                      --dataset='cifar10' \
                      --use_cpu=0
```

## Contact for Issues

*   Barret Zoph, @barretzoph <barretzoph@google.com>
*   Ekin Dogus Cubuk, <cubuk@google.com>
