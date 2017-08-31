%VL_NNNORMALIZE CNN Local Response Normalization (LRN)
%   Y = VL_NNORMALIZE(X, PARAM) computes the so-called Local Response
%   Normalization (LRN) operator. This operator performs a
%   channel-wise sliding window normalization of each column of the
%   input array X. The normalized output is given by:
%
%     Y(i,j,k) = X(i,j,k) / L(i,j,k)^BETA
%
%   where the normalization factor is given by
%
%     L(i,j,k) = KAPPA + ALPHA * (sum_{q in Q(k)} X(i,j,k)^2,
%
%   PARAM = [N KAPPA ALPHA BETA], and N is the size of the window. The
%   window Q(k) is defined as:
%
%     Q(k) = [max(1, k-FLOOR((N-1)/2)), min(D, k+CEIL((N-1)/2))].
%
%   where D is the number of feature channels in X. Note in particular
%   that, by setting N >= 2D, the function can be used to normalize
%   all the channels as a single group (useful to achieve L2
%   normalization).
%
%   DZDX = VL_NNORMALIZE(X, PARAM, DZDY) computes the derivative of
%   the block projected onto DZDY. DZDX and DZDY have the same
%   dimensions as X and Y respectively.
%
%   **Remark:** Some CNN libraries (e.g. Caffe) use a slightly
%   different convention for the parameters of the LRN. Caffe in
%   particular uses the convention:
%
%     PARAM_CAFFE = [N KAPPA N*ALPHA BETA]
%
%   i.e. the ALPHA paramter is multiplied by N.

% Copyright (C) 2014 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).
