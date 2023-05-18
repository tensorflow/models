# Masked Autoencoders Are Scalable Vision Learners (MAE)

TF2 implementation of [MAE](https://arxiv.org/abs/2111.06377).

## Imagenet pretrain

Model | reolution | pathch size | batch size | epochs | target pixel norm | val MSE
------------ | ------: | ------: | -----:| -----:| -----: | --------: 
(a) ViT-L14 | 224x224 | 14 | 4096 | 800 | no | 0.2456
(b) ViT-L14 | 224x224 | 14 | 4096 | 800 | yes | 0.3630
(c) ViT-L16 | 224x224 | 16 | 4096 | 800 | yes | 0.3866


## ImageNet linear probing

Model     | resolution | pathch size | base learning rate | batch size | init checkpoint | epochs | top1 Acc | dashboard
------------ | :--------: | -----:| -----:| -----:| -----:| -----:| -----: | --------:
ViT-L14 | 224x224 | 14 | 0.1 | 16384 | (b) | 90 | 72.8 | -
ViT-L16 | 224x224 | 16 | 0.1 | 16384 | (c) | 90 | 73.0 | -
ViT-L16 | 224x224 | 16 | 0.1 | 16384 | norm | 90 | 73.9 | Table 1 (d)

## ImageNet finetune

Model     | resolution | pathch size | base learning rate | batch size | init checkpoint | epochs | top1 Acc | dashboard
------------ | :--------: | -----:| -----:| -----:| -----:| -----:| -----: | -----:
ViT-L14 | 224x224 | 14 | 0.001 | 1024 | (a) | 50 | 84.4 | -
ViT-L14 | 224x224 | 14 | 0.001 | 1024 | (b) | 50 | 85.3 | -
ViT-L14 | 224x224 | 14 | 0.00075 | 1024 |(b) | 50 | 85.4 | -
ViT-L14 | 224x224 | 14 | 0.0001 | 4096| scratch | 200 | 82.4 | -
ViT-L16 | 224x224 | 16 | 0.001 | 1024 | (c)| 50 | 84.9 | -
ViT-L16 | 224x224 | 16 | 0.001 | 1024| no-norm | 50 | 84.9 | Table 1(d)
ViT-L16 | 224x224 | 16 | 0.001 | 1024| norm    | 50 | 85.4 | paper section 4.
ViT-L16 | 224x224 | 16 | 0.0001 | 4096| scratch | 200 | 82.5 | paper section 4.


## Known discrepancy with the paper:

*   ~-0.9 linear probing top1 acc (w/ norm) compared to paper results with patch
    size 16.

*   ~-0.5 finetune top1 acc (w/ norm) compared to paper results with patch
    size 16.
