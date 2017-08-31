% VL_NNROIPOOL  CNN region of interest pooling.
%   Y = VL_NNROIPOOL(X, ROIS) pools each feature channel in X in
%   the specified regions of interest ROIS. ROIS is a 5 x K array
%   containing K regions. Each region has five coordinates `[t, u0,
%   v0, u1, v1]` where `u0`, `v0` is the upper-left corner of a ROI,
%   `u1`, `v1` is the bottom-right corner, and `t` is the index of the
%   image that contains the region. Spatial coordiantes start at (1,1),
%   with `u` indexing the horizontal axis and `v` the vertical one.
%   The image indeces ranges from 1 to the number of images stored
%   in the tensor X.
%
%   If X has C feature channels, then the output Y is a 1 x 1 x C x K
%   array, with one image instance per region. Arguments can be SINGLE
%   or DOUBLE and CPU or GPU arrays; however, they must all be of the
%   same type (unless empty).
%
%   DZDX = VL_NNROIPOOL(X, ROIS, DZDY) computes the derivative of
%   the layer projected on DZDY with respect to X.
%
%   VL_NNROIPOOL(___, 'opt', value, ...) accepts the following
%   options:
%
%   `Method`:: `'max'`
%     Choose between `'max'` and `'avg'` (average) pooling.
%
%   `Subdivisions`:: `[1 1]`
%     Specifies the number [SH,SW] of vertical and horizontal tiles of
%     a region. This makes the output a SH x SW x C x K array.
%
%   `Transform`:: `1`
%     Specifies a spatial transformation to apply to region vertices before
%     they are applied to the input tensor. If T is a scalar, then
%     the transformation is a scaling centered at the origin:
%
%        u' = T (u - 1) + 1,
%        v' = T (v - 1) + 1.
%
%     If T is a 2D vector, then different scaling factors for the
%     `u` and `v` can be specified. Finally, if T is a 2 x 2 matrix, then:
%
%        u' = T(1,1) u + T(1,2) v + T(1,3),
%        v' = T(2,1) u + T(2,2) v + T(2,3).
%
%     Note that only the upper-left and bottom-right corners of each
%     rectangular region are transformed. Thus this is mostly useful
%     for axis-aligned transformations; the generality of the expression
%     allows, however, to swap `u` and `v`, which may be needed
%     to match different conventions for the box coordiantes.
%
%   See also: VL_NNPOOL().

% Copyright (C) 2016 Hakan Bilen, Abishek Dutta, and Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).
