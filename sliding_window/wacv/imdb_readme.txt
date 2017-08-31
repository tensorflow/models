% Mery, D.; Arteta, C.: Automatic Defect Recognition in X-ray Testing
% using Computer Vision. In 2017 IEEE Winter Conference on Applications of
% Computer Vision, WACV2017.
%
% Paper: http://dmery.sitios.ing.uc.cl/Prints/Conferences/International/2017-WACV.pdf
%
% (c) 2017 - Domingo Mery and Carlos Artera
%

% The 47.520 cropped images are stored in imdb.mat as follows
% imbd.images.label  47.520 x 1: 1 means defect, 2 is no-defect
% imbd.images.id     47.520 x 1: 1, 2, 3, ... 47520
% imbd.images.set    47.520 x 1: 1 = train, 2 = validation, 3 = test
% imbd.images.obj    47.520 x 1: series of GDXray, eg. 1 means series C00001
%          > cropped images from series 1 and 2 are used for testing
% imbd.images.images 32 x 32 x 47.520: 32 x 32 cropped images
% 
% Original X-ray images are from GDXray dataset, available on
% http://dmery.ing.puc.cl/index.php/material/gdxray/
