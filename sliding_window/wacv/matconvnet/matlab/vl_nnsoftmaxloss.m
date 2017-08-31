function y = vl_nnsoftmaxloss(x,c,dzdy)
%VL_NNSOFTMAXLOSS CNN combined softmax and logistic loss.
%   **Deprecated: use `vl_nnloss` instead**
%
%   Y = VL_NNSOFTMAX(X, C) applies the softmax operator followed by
%   the logistic loss the data X. X has dimension H x W x D x N,
%   packing N arrays of W x H D-dimensional vectors.
%
%   C contains the class labels, which should be integers in the range
%   1 to D. C can be an array with either N elements or with dimensions
%   H x W x 1 x N dimensions. In the fist case, a given class label is
%   applied at all spatial locations; in the second case, different
%   class labels can be specified for different locations.
%
%   DZDX = VL_NNSOFTMAXLOSS(X, C, DZDY) computes the derivative of the
%   block projected onto DZDY. DZDX and DZDY have the same dimensions
%   as X and Y respectively.

% Copyright (C) 2014-15 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

% work around a bug in MATLAB, where native cast() would slow
% progressively
if isa(x, 'gpuArray')
  switch classUnderlying(x) ;
    case 'single', cast = @(z) single(z) ;
    case 'double', cast = @(z) double(z) ;
  end
else
  switch class(x)
    case 'single', cast = @(z) single(z) ;
    case 'double', cast = @(z) double(z) ;
  end
end

%X = X + 1e-6 ;
sz = [size(x,1) size(x,2) size(x,3) size(x,4)] ;

if numel(c) == sz(4)
  % one label per image
  c = reshape(c, [1 1 1 sz(4)]) ;
end
if size(c,1) == 1 & size(c,2) == 1
  c = repmat(c, [sz(1) sz(2)]) ;
end

% one label per spatial location
sz_ = [size(c,1) size(c,2) size(c,3) size(c,4)] ;
assert(isequal(sz_, [sz(1) sz(2) sz_(3) sz(4)])) ;
assert(sz_(3)==1 | sz_(3)==2) ;

% class c = 0 skips a spatial location
mass = cast(c(:,:,1,:) > 0) ;
if sz_(3) == 2
  % the second channel of c (if present) is used as weights
  mass = mass .* c(:,:,2,:) ;
  c(:,:,2,:) = [] ;
end

% convert to indexes
c = c - 1 ;
c_ = 0:numel(c)-1 ;
c_ = 1 + ...
  mod(c_, sz(1)*sz(2)) + ...
  (sz(1)*sz(2)) * max(c(:), 0)' + ...
  (sz(1)*sz(2)*sz(3)) * floor(c_/(sz(1)*sz(2))) ;

% compute softmaxloss
xmax = max(x,[],3) ;
ex = exp(bsxfun(@minus, x, xmax)) ;

%n = sz(1)*sz(2) ;
if nargin <= 2
  t = xmax + log(sum(ex,3)) - reshape(x(c_), [sz(1:2) 1 sz(4)]) ;
  y = sum(sum(sum(mass .* t,1),2),4) ;
else
  y = bsxfun(@rdivide, ex, sum(ex,3)) ;
  y(c_) = y(c_) - 1;
  y = bsxfun(@times, y, bsxfun(@times, mass, dzdy)) ;
end
