% [T,p] = wacv_demo(fxname,clname,clparameter)
%
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
% Input Parameters:
% fxname      : name of the features ('clp''bsif','txh','gabor','gaborfull','int','slbp','src','alx','ggl','vgg1','vgg3','vgg4')
%              'int'         - Intensity features (grayvalues)
%              'lbp'         - Local Binary Patterns (59 bins)
%              'lbpri'       - Rotation invariant LBP (36 bins)
%              'slbp'        - Semantic LBP
%              'clp'         - Crossing line profile (CLP)
%              'txh'         - Haralick texture feature
%              'fft'         - Discrete Fourier transform
%              'dct'         - Discrete cosine transform
%              'gabor'       - Gabor features
%              'gabor+'      - Gabor plus features
%              'bsif'        - Binarized statistical image features (BSIF)
%              'hog'         - Histogram of orientated gradients
%              'surf'        - Speeded up robust feature (SURF)
%              'sift'        - Scale invariant feature transform (SIFT)
%              'brisk'       - Binary robust invariant scalable keypoint (BRISK)
%              'alex'        - AlexNet (imagenet-caffe-alex.mat)
%              'ggl'         - GoogleNet (imagenet-googlenet-dag.mat)
%              'vgg1'        - VGG-F (imagenet-vgg-f.mat)
%              'vgg2'        - VGG-very-deep-16 (imagenet-vgg-verydeep-16.mat)
%              'vgg3'        - VGG-very-deep-19 (imagenet-vgg-verydeep-19.mat)
%              'vgg4'        - VGG-M-2048 (imagenet-vgg-m-2048.mat)
% 
% clname      : name of the classifier ('knn','libsvm','ann')
% clparameter : parameter of the classifier
%
% Output Parameters:
%   info.Xtrain  training features
%   info.Xtest   testing features
%   info.dtrain  training labels
%   info.dtest   testing labels
%   info.opts    parameters of the learned classifier
%   info.ds      prediction on the testing data
%   info.T       confusion matrix = [ TP FP; FN TN ]
%   info.acc     accuracy = (TP+TN)/(TP+FP+FN+TN)
%   info.opfx    parameters of the features
%   info.opcl    parameters of the classifier
%
% Example 1:
% info  = wacv_demo('lbpri','ann',15); % LBP-ri features and an Artifical
%                                      % Neural Network with 15 hidden layers
%
% Example 2:
% info  = wacv_demo('surf','knn',5);   % SURF features and an KNN classifier
%                                      % with 5 neighbours
%
% Example 3:
% info  = wacv_demo('src','src',10);   % SRC features (intensity) and SRC classifier
%                                      % with 10 atoms in the sparse representation
%                                      % warning: more than 4 hours!
% Example 4:
% info  = wacv_demo('bsif','libsvm','-t 0'); % BSIF features and
%                                      % Linear SVM classifier
%
%
% Note: The results might be a little bit different from those presented in
% Table 3 of the paper because of some random procedures (eg, ANN initialization) 

function info = wacv_demo(fxname,clname,clparameter)

fprintf('\nWACV-Experiments for Defects Detection\n');
fprintf('D. Mery and C. Arteta: Automatic Defect Recognition in X-ray Testing \nusing Computer Vision (WACV2017)\n\n');

fprintf('Loading imdb.mat ...\n\n');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load cropped images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
f                = 'imdb.mat';
load(f)

% The 47.520 cropped images are stored in cflaws.mat as follows
% imbd.images.label  47.520 x 1: 1 means defect, 2 is no-defect
% imbd.images.id     47.520 x 1: 1, 2, 3, ... 47520
% imbd.images.set    47.520 x 1: 1 = train, 2 = validation, 3 = test
% imbd.images.obj    47.520 x 1: series of GDXray, eg. 1 means series C00001
%          > cropped images from series 1 and 2 are used for testing
% imbd.images.images 32 x 32 x 47.520: 32 x 32 cropped images


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Training, validation and testing images
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
subsampling = 1; % cropped images are not subsampled
im1         = imdb.images.data(:,:,1);    % a sample
i1          = find(imdb.images.set<=2)';  % training and validation
i2          = find(imdb.images.set==3)';  % testing
i1          = i1(1:subsampling:end);
i2          = i2(1:subsampling:end);
ix_train    = [imdb.images.label(i1)' i1];
ix_test     = [imdb.images.label(i2)' i2];
imageMean   = mean(imdb.images.data(:)) ;

clear imdb

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Definition of features and classifier
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Features   = %s\n',fxname);
opfx = wacv_fxdef(fxname,im1);
opfx.imageMean = imageMean;
if ischar(clparameter)
    clparst = clparameter;
else
    clparst = num2str(clparameter);
end
fprintf('Classifier = %s-%s\n\n',clname,clparst);
opcl = wacv_cldef(clname,clparameter);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Extraction of training features
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Extracting %s features ...\n',fxname);
[Xtrain,dtrain] = exp_fx('wacv_fx',f,opfx,ix_train,[ fxname ': training features']);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Extraction of testing features
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[Xtest,dtest] = exp_fx('wacv_fx',f,opfx,ix_test,[fxname ': testing features']);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Training
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Training %s-%s classifier ...\n',clname,clparst);
opts = exp_train('wacv_classifier',Xtrain,dtrain,opcl);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Testing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Testing ...\n');
ds = exp_test('wacv_test',Xtest,opts);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Evaluation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
C = Bev_confusion(ds,dtest);   % Confusion Matrix
p = Bev_performance(ds,dtest); % Accuracy

fprintf('TP       = %d\n',C(1,1));
fprintf('FP       = %d\n',C(1,2));
fprintf('TN       = %d\n',C(2,2));
fprintf('FN       = %d\n',C(2,1));
fprintf('Accuracy = %5.2f%%\n\n',p*100);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Info
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

info.Xtrain = Xtrain;   % training features
info.Xtest  = Xtest;    % testing features
info.dtrain = dtrain;   % training labels
info.dtest  = dtest;    % testing labels
info.opts   = opts;     % parameters of the learned classifier
info.ds     = ds;       % prediction on the testing data
info.C      = C;        % confusion matrix
info.acc    = p;        % accuracy
info.opfx   = opfx;     % parameters of the features
info.opcl   = opcl;     % parameters of the classifier
