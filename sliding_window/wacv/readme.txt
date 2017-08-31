% Mery, D.; Arteta, C.: Automatic Defect Recognition in X-ray Testing
% using Computer Vision. In 2017 IEEE Winter Conference on Applications of
% Computer Vision, WACV2017.
%
% Paper: http://dmery.sitios.ing.uc.cl/Prints/Conferences/International/2017-WACV.pdf
%
% (c) 2017 - Domingo Mery and Carlos Artera
%
% This code needs the following toolboxes:
% - MatConvNet      > http://www.vlfeat.org
% - VLFeat          > http://www.vlfeat.org
% - SPAMS           > http://spams-devel.gforge.inria.fr
% - Neural Networks > http://www.mathworks.com
% - Computer Vision > http://www.mathworks.com
% - LIBSVM          > http://www.csie.ntu.edu.tw/ cjlin/libsvm
% - Balu            > http://dmery.ing.puc.cl/index.php/balu
%
% Original images are from GDXray
% > http://dmery.ing.puc.cl/index.php/material/gdxray/
%
% 0. Dataset:
%    > see imdb.mat and imdb_readme.txt
%
% 1. Code for classical feature extraction and classification:
%    > see wacv_demo.m and wacv_examples.m
%    > in this example the following pretrained nets must be
%    > included in folder nets:
%    > imagenet-caffe-alex.mat		imagenet-vgg-f.mat          imagenet-vgg-verydeep-16.mat
%    > imagenet-googlenet-dag.mat	imagenet-vgg-m-2048.mat		imagenet-vgg-verydeep-19.mat
%    > please see vlfeat.org for downloading these files.
%
% 2. Code fox Xnet: 
%    > see folder xnet and use code xnet_main.m for training and testing.
%
% 3. Sliding windows for detection in one X-ray image:
%    > see wacv_sliwin.m
