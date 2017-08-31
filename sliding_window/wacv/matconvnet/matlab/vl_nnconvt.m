%VL_NNCONVT CNN convolution transpose.
%   Y = VL_NNCONVT(X, F, B) computes the transposed convolution of
%   the image stack X with the filter bank F and biases B. If B is
%   the empty matrix, then no biases are added.
%
%   X is a SINGLE array of dimension H x W x D x N where (H,W) are
%   the height and width of the image stack, D is the image depth
%   (number of feature channels) and N the number of of images in the
%   stack.
%
%   F is a SINGLE array of dimension FW x FH x K x FD where (FH,FW)
%   are the filter height and width, K the number of filters in the
%   bank, and FD the depth of a filter (the same as the depth of
%   image X). Filter k is givenby elements F(:,:,k,:); this differ
%   from VL_NNCONV() where a filter is given by elements
%   F(:,:,:,k). FD must be the same as the input depth D.
%
%   B is a SINGLE array with 1 x 1 x K elements (B can in fact
%   be of any shape provided that it has K elements).
%
%   [DZDX, DZDF, DZDB] = VL_NNCONVT(X, F, B, DZDY) computes the
%   derivatives of the block projected onto DZDY. DZDX, DZDF, DZDB,
%   and DZDY have the same dimensions as X, F, B, and Y
%   respectively. In particular, if B is the empty matrix, then DZDB
%   is also empty.
%
%   VL_NNCONVT(..., 'option', value, ...) takes the following options:
%
%   `Upsample`:: 1
%     The input stride (upsampling factor). Passing [UPY UPX] allows
%     specifying different upsampling factors for the vertical and
%     horizontal directions.
%
%   `Crop`:: 0
%     The amount of output cropping. [TOP BOTTOM LEFT RIGHT] pixels
%     around the output image are dropped. Passing a scalar applies
%     the same amount of crop to all borders.
%
%   `NumGroups`:: 1
%     The number of filter groups. This parameter allows using filter
%     groups in the same way as defined by VL_NNCONV(). NUMGROUPS
%     must divide the filter bank depth FD. In this case, filters are
%     divided in NUMGROUPS different groups, each operating on a
%     equal number of contiguous dimensions of the input. FILTERS is
%     then interpreted as containing K * NUMGROUPS different filters,
%     each of depth FD / NUMGROUPS.
%
%   ## About the convolution transpose operator
%
%   The convolution transpose operator is defined as follows. Let U =
%   VL_NNCONV(V, F, []). Since this is a linear operation, there is a
%   matrix M such that U(:) = M V(:). The convolution transpose is
%   the linear convolution operator that results in Y(:) = M'
%   X(:). See the PDF manual for further detials.
%
%   There are two main uses for this operator. As a sort of 'reverse'
%   of convolution, useful for example in a deconvolutional network,
%   and as an interpolating filter (instead of a decimating one).
%
%   The output a is a SINGLE array of dimension YH x YW x K x N of N
%   images with K channels and size:
%
%     YH = UPH (XH - 1) + FH - CROPTOP - CROPBOTTOM,
%     YW = UPW (XW - 1) + FW - CROPLEFT - CROPRIGHT.
%
%   ## CUDNN SUPPORT
%
%   If compiled in, the function will use cuDNN convolution routines
%   (with the exception of asymmetric left-right or top-bottom
%   padding and a few corner cases such as 1x1 filters in Linux that
%   trigger current bugs in cuDNN). You can use the 'NoCuDNN' option
%   to disable cuDNN or 'cuDNN' to activate it back again (the choice
%   sticks until MATLAB purges the MEX files for any reason).
%
%   Some CuDNN algorithms may use a very large amount of memory on the
%   GPU (workspace). MatConvNet requests CuDNN to use at most 512MB of
%   GPU memory for the workspace. To change this behaviour, use the
%   `CudnnWorskpaceLimit` option to specify the maximum size of the
%   workspace in bytes. Set this parameter +inf to remove the limit
%   and use the `Verbose` flag to check how much memory is being used.


% Copyright (C) 2015 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).
