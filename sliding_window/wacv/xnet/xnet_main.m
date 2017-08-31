% xnet_main.m
%
% CNN training and testing for Xnet.
%
% Mery, D.; Arteta, C.: Automatic Defect Recognition in X-ray Testing
% using Computer Vision. In 2017 IEEE Winter Conference on Applications of
% Computer Vision, WACV2017.
%
% Paper: http://dmery.sitios.ing.uc.cl/Prints/Conferences/International/2017-WACV.pdf
%
% (c) 2017 - Domingo Mery and Carlos Artera

clc

fprintf('\nWACV-Experiments for Defects Detection\n');
fprintf('D. Mery and C. Arteta: Automatic Defect Recognition in X-ray Testing \nusing Computer Vision (WACV2017)\n\n');

fprintf('\nXnet: Training and Testing\n');


epochs = 500;

x = [32 64 128 64 7 5 3 2]; % see Table 2 of paper

param.a  = x(1);
param.b  = x(2);
param.c  = x(3);
param.d  = x(4);
param.p1 = x(5);
param.p2 = x(6);
param.p3 = x(7);
param.p4 = x(8);


fprintf('Training Xnet with %d epochs...\n',epochs);
[net, info] = xnet_cnn(param,epochs,'train');
disp('Testing Xnet...');
[net, info] = xnet_cnn(net,info,'test');


figure; clf; colormap jet;vl_imarraysc(squeeze(net.params(1).value),'spacing',2)
title('filters')
C = info.C;
p = info.p;
fprintf('TP       = %d\n',C(1,1));
fprintf('FP       = %d\n',C(1,2));
fprintf('TN       = %d\n',C(2,2));
fprintf('FN       = %d\n',C(2,1));
fprintf('Accuracy = %5.2f%%\n\n',p*100);



