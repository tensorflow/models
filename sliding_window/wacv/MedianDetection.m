function R = MedianDetection(I,params)
Y0 = Ximgaussian(I,params.gaussianmask);
Y1 = Ximmedian(Y0,params.medianmask);
Y2 = (double(Y0)-double(Y1))>params.threshold;
Y3 = bwareaopen(Y2,params.areamin);
Y4  = imclearborder(Y3);
se = strel('disk',round((params.dilationmask-1)/2));
R  = imdilate(Y4,se);
R  = imclearborder(R);
subplot(2,2,1);imshow(Y1,[]); title('Original');
subplot(2,2,2);imshow(Y2,[]); title('Saliency spots');
subplot(2,2,3);imshow(Y4,[]); title('Large spots');
subplot(2,2,4);imshow(R,[]); title('Eroded spots');
