function y = vl_nnnoffset(x, param, dzdy)
%VL_NNNOFFSET CNN norm-dependent offset.
%   Y = VL_NNNOFFSET(X, PARAM) subtracts from each element of X the
%   weighted norm of the feature channels:
%
%     X(i,j,k) = X(i,j,k) - PARAM(1) * L(i,j) ^ PARAM(2)
%
%   where
%
%     L(i,j) = sum_K X(i,j,k)^2
%
%   DZDX = VL_NNNOFFSET(X, PARAM, DZDY) computes the derivative of the
%   block projected onto DZDY. DZDX and DZDY have the same dimensions
%   as X and Y respectively.

% Copyright (C) 2014 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

L = sum(x.^2,3) ;
L = max(L, 1e-8) ;

if nargin <= 2
  y = bsxfun(@minus, x, param(1)*L.^param(2)) ;
else
  y = dzdy - bsxfun(@times, (2*param(1)*param(2))* x, sum(dzdy,3) .* (L.^(param(2)-1))) ;
end
