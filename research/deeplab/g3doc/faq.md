# FAQ
___
Q1: What if I want to use other network backbones, such as ResNet [1], instead of only those provided ones (e.g., Xception)?

A: The users could modify the provided core/feature_extractor.py to support more network backbones.
___
Q2: What if I want to train the model on other datasets?

A: The users could modify the provided dataset/build_{cityscapes,voc2012}_data.py and dataset/segmentation_dataset.py to build their own dataset.
___
Q3: Where can I download the PASCAL VOC augmented training set?

A: The PASCAL VOC augmented training set is provided by Bharath Hariharan et al. [2] Please refer to their [website](http://home.bharathh.info/pubs/codes/SBD/download.html) for details and consider citing their paper if using the dataset.
___
Q4: Why the implementation does not include DenseCRF [3]?

A: We have not tried this. The interested users could take a look at Philipp Krähenbühl's [website](http://graphics.stanford.edu/projects/densecrf/) and [paper](https://arxiv.org/abs/1210.5644) for details.
___
Q5: What if I want to train the model and fine-tune the batch normalization parameters?

A: Fine-tuning batch normalization requires large batch size, and thus in the train.py we suggest setting `num_clones` (number of GPUs on one machine) and `train_batch_size` to be as large as possible.
___
Q6: How can I train the model asynchronously?

A: In the train.py, the users could set `num_replicas` (number of machines for training) and `num_ps_tasks` (we usually set `num_ps_tasks` = `num_replicas` / 2). See slim.deployment.model_deploy for more details.
___
## References

1. **Deep Residual Learning for Image Recognition**<br />
   Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun<br />
   [[link]](https://arxiv.org/abs/1512.03385), In CVPR, 2016.

2. **Semantic Contours from Inverse Detectors**<br />
   Bharath Hariharan, Pablo Arbelaez, Lubomir Bourdev, Subhransu Maji, Jitendra Malik<br />
   [[link]](http://home.bharathh.info/pubs/codes/SBD/download.html), In ICCV, 2011.

3. **Efficient Inference in Fully Connected CRFs with Gaussian Edge Potentials**<br />
   Philipp Krähenbühl, Vladlen Koltun<br />
   [[link]](http://graphics.stanford.edu/projects/densecrf/), In NIPS, 2011.
