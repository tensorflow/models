% Mery, D.; Arteta, C.: Automatic Defect Recognition in X-ray Testing
% using Computer Vision. In 2017 IEEE Winter Conference on Applications of
% Computer Vision, WACV2017.
%
% Paper: http://dmery.sitios.ing.uc.cl/Prints/Conferences/International/2017-WACV.pdf
%
% (c) 2017 - Domingo Mery and Carlos Artera

% Sliding window example for Figure 5
clt
I0 = imread('casting.png');
[M0,N0] = size(I0);

H = fspecial('gaussian',32,6);
H = H/max2(H);

d = 3;

s = 1;

HM0 = zeros(size(I0));

M = 32; N = 32;
N2 = round(N/2);
M2 = round(M/2);

fxname = 'lbp';
load opts % trained classifier (LIBSVM linear kernel using LBP features)

figure(1)
param.gaussianmask = round(s*4);
param.medianmask   = round(s*8+1);
param.threshold    = param.medianmask/3-1;
param.areamin      = round(s*16);
param.dilationmask = param.gaussianmask+1;

% saliency detection, only in this pixels defects will be searched
K0 = MedianDetection(I0,param);
figure(2)
Bio_edgeview(I0,K0)
title('Preselected pixels');


figure(3)
I = single(double(imresize(I0,s))/255);
K = imresize(K0,s,'nearest');
imshow(I,[]);
[MM,NN] = size(I);
im = zeros(M,N);
ft = Bio_statusbar('extracting');
opfx             = wacv_fxdef(fxname,im);
HM = zeros(MM,NN);
for i=1:d:MM-M
    ft = Bio_statusbar(i/MM,ft);
    i1 = i; i2 = i1+M-1;
    for j=1:d:NN-N
        j1 = j; j2 = j1+N-1;
        if K(i1+M2,j1+N2)==1
            x = Bfx_lbp(I(i1:i2,j1:j2),opfx);
            ds = exp_test('wacv_test',x,opts);
            if ds==1
                HM(i1:i2,j1:j2) = HM(i1:i2,j1:j2)+H;
            end
        end
    end
    imshow(HM/(max2(HM)+0.0001)*256,jet)
    drawnow
end
delete(ft);

D = HM>4;
figure(4)
Bio_edgeview(I,bwperim(D))
title('First detection');

HM0 = HM0 + imresize(HM,[M0 N0]);
figure(5)
imshow(HM0/max2(HM0)*64,jet)
title('Heat map')
figure(6)
Bio_edgeview(I0,bwperim(HM0>(max2(HM0)*0.5)))
figure(7)
D = HM0>10;
Bio_edgeview(I0,bwperim(D))

% remove small and large regions
[L,n] = bwlabel(D);
R = zeros(size(D));
for i=1:n
    A = sum2(L==i);
    if and(A>25,A<600)
        R = or(R,L==i);
    end
end

figure(8)
Bio_edgeview(I0,bwperim(R))
title('Final detection');
