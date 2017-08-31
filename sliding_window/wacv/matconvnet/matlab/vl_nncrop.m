function y = vl_nncrop(x, crop, dzdy, inputSize)
%VL_NNCROP CNN crop.
%   Y = VL_NNCROP(X, CROP) crops the input X spatially. CROP specifies the
%   amount of cropping as [TOP, BOTTOM, LEFT, RIGHT].
%
%   DZDX = VL_NNCROP(X, CROP, DZDY) computes the derivative DZDX of the
%   function projected on the output derivative DZDY. DZDX has the same
%   dimension as X and DZDY the same dimension as Y.
%
%   DZDX = VL_NNCROP([], CROP, DZDY, INPUTSIZE) is an alternative to
%   the previous call in which X is omitted and its size is passed as
%   INPUTSIZE.

% Copyright (C) 2015 Sebastien Ehrhardt and Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

if nargin < 4
  sz = [size(x,1) size(x,2) size(x,3) size(x,4)] ;
else
  sz = inputSize ;
end

sv = 1 + crop(1) : sz(1) - crop(2) ;
su = 1 + crop(3) : sz(2) - crop(4) ;

if nargin <= 2 || isempty(dzdy)
  y = x(sv, su, :, :) ;
else
  if isa(dzdy, 'gpuArray')
    y = gpuArray.zeros(sz, classUnderlying(dzdy)) ;
  else
    y = zeros(sz, class(dzdy)) ;
  end
  y(sv, su, :, :) = dzdy ;
end
