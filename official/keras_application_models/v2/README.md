# Scripts for benchmarking keras application models on V2

## Train ResNet50 on CIFAR-10 with pretrained ImageNet weights

Architecture:

Pretrained ResNet50 without top layer, but a manually added softmax FC for classification.

Optimization:

+ Normalization:
  * pixel mean subtracted
+ Data augmentation:
  * 3 pixel random crop, with zero padding
  * random horizontal flip
+ SGD Optimizer, with 0.9 momentum
+ Learning rate schedule:
  * Epoch 1-10: 1e-3
  * Epoch 11-30: 1e-4
  * Epoch 30+: 1e-5
+ L2 regularization: 1e-4

Command:

```
python3 train_resnet50.py --benchmark_log_dir=/tmp/metric_dir --num_gpus=1 --train_epochs=200 --batch_size=32
```

Could reach 85%+ test accuracy in 20 epoches.


