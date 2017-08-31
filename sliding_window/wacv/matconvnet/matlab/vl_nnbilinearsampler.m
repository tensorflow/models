%VL_NNBILIEARSAMPLER  CNN spatial bilinear resampling
%   Y = VL_NNBILINEARSAMPLER(X,GRID) resamples image X at the spatial
%   locations specified by GRID using bilinear interpolation.
%
%   X is a array of dimension H x W x C x N, where (H,W) are the
%   height and width of the image, C is the number of feature
%   channels, and N is the number of images in the batch.
%
%   GRID is an array of dimension 2 x Ho x Wo x No, where (Ho,Wo) are
%   the height and width of the output image and No the number of
%   output images in the output batch Y. The output array Y has
%   dimensions Ho x Wo x C x No. The same resampling grid is used for
%   all input feature channels, but each output image in the batchY
%   uses its own grid.
%
%   For output image n, GRID(1,:,:,n) specifies the vertical location
%   v of a sample in the input image X and GRID(2,:,:,n) the
%   horizontal location u. The convention follows standard
%   impelemntations of this operator in the literature. Namely:
%
%   1. The grid coordinates are normalized in the range [-1,1]. This
%      means that (-1,-1) is the center of the upper-left pixel in the
%      input image and (+1,+1) the center of the bottom-right pixel.
%
%   2. The V,U coordiante planes are stacked in the fisrt dimension of
%      GRID instead of in the third, as it would be more natural in
%      MatConvNet (as these could be interpreted as 'channels' in
%      GRID).
%
%   Further, No can be a multiple of N; in this case, it is assumed
%   that there are No/N transforms per input image, hence, the
%   transforms [1 ... No/N] are applied to the first image, [No/N+1
%   ... 2*No/N] are applied to the second image, etc.
%
%   [DX, DGRID] = VL_NNBILINEARSAMPLER(X, GRID, DY) computes the
%   derivatives of the block projected onto DY. DX, DGRID, DY have the
%   same dimensions as X, GRID and Y, respectively.
%
%   ## CUDNN SUPPORT
%
%   If compiled in, the function will use cuDNN's
%   implementation. Note, cuDNN v5 or higher is required.
%   You can use the 'NoCudnn' option to disable
%   cuDNN or 'CuDNN' to activate it back again (the
%   choice sticks until MATLAB purges the MEX files for any reason).

% Copyright (C) 2016 Ankush Gupta and Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).
