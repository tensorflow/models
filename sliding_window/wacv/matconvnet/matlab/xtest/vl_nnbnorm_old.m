function [y,dzdg,dzdb] = vl_nnbnorm_old(x,g,b,varargin)
% VL_NNBNORM  CNN batch normalisation
%   Y = VL_NNBNORM(X,G,B) computes the batch normalization of the
%   input X. This is defined as:
%
%      Y(i,j,k,t) = G(k) * (X(i,j,k,t) - mu(k)) / sigma(k) + B(k)
%
%   where
%
%      mu(k) = mean_ijt X(i,j,k,t),
%      sigma(k) = sqrt(sigma2(k) + EPSILON),
%      sigma2(k) = mean_ijt (X(i,j,k,t) - mu(k))^2
%
%   are respectively the per-channel mean, standard deviation, and
%   variance of the input and G(k) and B(k) define respectively a
%   multiplicative and additive constant to scale each input
%   channel. Note that statistics are computed across all feature maps
%   in the batch packed in the 4D tensor X. Note also that the
%   constant EPSILON is used to regularize the computation of sigma(k)
%
%   [Y,DZDG,DZDB] = VL_NNBNORM(X,G,B,DZDY) computes the derviatives of
%   the output Z of the network given the derivatives with respect to
%   the output Y of this function.
%
%   VL_NNBNROM(..., 'Option', value) takes the following options:
%
%   `Epsilon`:: 1e-4
%       Specify the EPSILON constant.
%
%   See also: VL_NNNORMALIZE().

% Copyright (C) 2015 Karel Lenc and Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

% ISSUE - needs to store internal state, another reason for having classes?

% -------------------------------------------------------------------------
%                                                             Parse options
% -------------------------------------------------------------------------

opts.epsilon = 1e-4 ;
backMode = numel(varargin) > 0 && ~ischar(varargin{1}) ;
if backMode
  dzdy = varargin{1} ;
  opts = vl_argparse(opts, varargin(2:end)) ;
else
  opts = vl_argparse(opts, varargin) ;
end

% -------------------------------------------------------------------------
%                                                                    Do job
% -------------------------------------------------------------------------

x_size = [size(x,1), size(x,2), size(x,3), size(x,4)] ;
g_size = size(g) ;
b_size = size(b) ;
g = reshape(g, [1 x_size(3) 1]) ;
b = reshape(b, [1 x_size(3) 1]) ;
x = reshape(x, [x_size(1)*x_size(2) x_size(3) x_size(4)]) ;

mass = prod(x_size([1 2 4])) ;
mu = sum(sum(x,1),3) / mass  ;
y = bsxfun(@minus, x, mu); % y <- x_mu
sigma2 = sum(sum(y .* y,1),3) / mass + opts.epsilon ;
sigma = sqrt(sigma2) ;

if ~backMode
  y = bsxfun(@plus, bsxfun(@times, g ./ sigma, y), b) ;
else
  % remember: y contains x_mu
  dzdy = reshape(dzdy, size(x)) ;
  dzdg = sum(sum(dzdy .* y,1),3) ./ sigma ;
  dzdb = sum(sum(dzdy,1),3) ;

  muz = dzdb / mass;
  y = ...
    bsxfun(@times, g ./ sigma, bsxfun(@minus, dzdy, muz)) - ...
    bsxfun(@times, g .* dzdg ./ (sigma2 * mass), y) ;

  dzdg = reshape(dzdg, g_size) ;
  dzdb = reshape(dzdb, b_size) ;
end

y = reshape(y, x_size) ;
end
