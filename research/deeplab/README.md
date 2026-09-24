# Welcome to my fork of official DeepLab repository
I uploaded my new files and changes of official DeepLab files here. For official DeepLab README please see README_original.md <br />

This repository contains my bachelor thesis work. The aim of the thesis was to make a semantic segmentation model which could recognize various types of surfaces in nature. The robots of BUT's researchers (https://www.vut.cz/en/) maybe will use my trained model to identify if the robot can get through some type of terrain.<br />

To train the model, I used RUGD dataset (http://rugd.vision/).<br />
To see some segmented examples of RUGD dataset with trained model, please go to datasets/rugd/exp/train_on_trainval_set/vis/segmentation_results/<br />
After some small changes (comment or uncomment) in test_video.py and test_video2.py you can try trained model with your own images or video.<br />
video_z_datasetu.mp4 is video of test ride with BUT's robot. <br />
If you want to use RUGD dataset to train the model with 8 bit images, you can use color palette RUGD_vsetky_farby.pal with add_palete.bat to have all images with the same colormap.<br />
vis.py is my modified version of vis_orig.py(which is vis.py but renamed).<br />
There are some new files in utils repository - get_dataset_colormap.py and save_annotation.py. They contain code to work with RUGD dataset.<br />
In datasets, there are 2 modified files - data_generator.py and build_voc2012_data_Vlastne_obrazky.py - which are used to make TFrecord of RUGD.

