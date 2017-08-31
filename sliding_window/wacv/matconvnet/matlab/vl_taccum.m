function a = vl_taccum(alpha, a, beta, b)
%VL_TACCUM  Compute A = alpha A + beta B
%   A = VL_TACCUM(ALPHA, A, BETA, B) computes efficiently A = alpha A
%   + beta B. For GPU arrays, it performs its computation in place, by
%   modifiying A without creating an additional copy.

% Copyright (C) 2016 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

if isscalar(a) || isscalar(b)
  a = alpha * a + beta * b ;
  return ;
elseif isa(a, 'gpuArray')
  vl_taccummex(alpha, a, beta, b, 'inplace') ;
else
  a = vl_taccummex(alpha, a, beta, b) ;
end
