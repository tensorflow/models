# Masked Autoencoders Are Scalable Vision Learners (MAE)

TF2 implementation of [MAE](https://arxiv.org/abs/2111.06377).

## Known discrepancy with the paper:

*   ~-0.9 linear probing top1 acc (w/ norm) compared to paper results with patch
    size 16.

*   ~-0.5 finetune top1 acc (w/ norm) compared to paper results with patch
    size 16.
