function [y1, y2] = vl_nnpdist(x, x0, p, varargin)
%VL_NNPDIST CNN p-distance from target.
%   VL_NNPDIST(X, X0, P) computes the P distance raised of each feature
%   vector in X to the corresponding feature vector in X0:
%
%     Y(i,j,1) = (SUM_d (X(i,j,d) - X0(i,j,d))^P)^(1/P)
%
%   X0 should have the same size as X; the outoput Y has the same
%   height and width as X, but depth equal to 1. Optionally, X0 can
%   be a 1 x 1 x D x N array, in which case the same target feature
%   vector in X0 is compared to all feature vectors in X. In that case,
%   however, the DZDX0 are of size of X.
%
%   Setting the `noRoot` option to `true` does not take the 1/P power
%   in the formula, computing instead
%
%     Y(i,j,1) = SUM_d (X(i,j,d) - X0(i,j,d))^P
%
%   For example, `vl_nnpdist(x, x0, 2, 'noRoot', true)` computes the
%   squared L2 distance.
%
%   [DZDX, DZDX0] = VL_NNPDISTP(X, X0, P, DZDY) computes the derivative
%   of the block inputs projected onto DZDY. DZDX, DZDX0 and DZDY have the
%   same dimensions as X and Y, respectively.
%
%   VL_NNPDIST(___, 'OPT', VAL, ...) accepts the following options:
%
%   `NoRoot`:: `false`
%      If set to true, compute the P-distance to the P-th power.
%
%   `Epsilon`:: 1e-6
%      When computing derivatives, quantities that are divided in are
%      lower boudned by this value. For example, the L2 distance is
%      not smooth at the origin; this option prevents the
%      derivative from diverging.
%
%   `Aggregate`:: false
%      Instead of returning one scalar for each spatial location in
%      the inputs, sum all of them into a single scalar.
%
%   `InstanceWeights`:: `[]`
%      Optionally weight individual instances. This parameter can be
%      eigther a scalar or a weight mask, one for each pixel in the
%      input tensor.

% Copyright (C) 2015  Karel Lenc and Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

% -------------------------------------------------------------------------
%                                                             Parse options
% -------------------------------------------------------------------------

opts.noRoot = false ;
opts.epsilon = 1e-6 ;
opts.aggregate = false ;
opts.instanceWeights = [] ;
backMode = numel(varargin) > 0 && ~ischar(varargin{1}) ;
if backMode
  dzdy = varargin{1} ;
  opts = vl_argparse(opts, varargin(2:end), 'nonrecursive') ;
else
  dzdy = [] ;
  opts = vl_argparse(opts, varargin, 'nonrecursive') ;
end

% -------------------------------------------------------------------------
%                                                             Parse options
% -------------------------------------------------------------------------

d = bsxfun(@minus, x, x0) ;

if ~isempty(dzdy) && ~isempty(opts.instanceWeights)
  dzdy = bsxfun(@times, opts.instanceWeights, dzdy) ;
end

if ~opts.noRoot
  if isempty(dzdy)
    if p == 1
      y1 = sum(abs(d),3) ;
    elseif p == 2
      y1 = sqrt(sum(d.*d,3)) ;
    else
      y1 = sum(abs(d).^p,3).^(1/p) ;
    end
  else
    if p == 1
      y1 = bsxfun(@times, dzdy, sign(d)) ;
    elseif p == 2
      y1 = max(sum(d.*d,3), opts.epsilon).^(-0.5) ;
      y1 = bsxfun(@times, bsxfun(@times, dzdy, y1),  d) ;
    elseif p < 1
      y1 = sum(abs(d).^p,3).^((1-p)/p) ;
      y1 = bsxfun(@times, bsxfun(@times, dzdy, y1), max(abs(d), opts.epsilon).^(p-1) .* sign(d)) ;
    else
      y1 = max(sum(abs(d).^p,3), opts.epsilon).^((1-p)/p) ;
      y1 = bsxfun(@times, bsxfun(@times, dzdy, y1), abs(d).^(p-1) .* sign(d)) ;
    end
  end
else
  if isempty(dzdy)
    if p == 1
      y1 = sum(abs(d),3) ;
    elseif p == 2
      y1 = sum(d.*d,3) ;
    else
      y1 = sum(abs(d).^p,3) ;
    end
  else
    if p == 1
      y1 = bsxfun(@times, dzdy, sign(d)) ;
    elseif p == 2
      y1 = bsxfun(@times, 2 * dzdy, d) ;
    elseif p < 1
      y1 = bsxfun(@times, p * dzdy, max(abs(d), opts.epsilon).^(p-1) .* sign(d)) ;
    else
      y1 = bsxfun(@times, p * dzdy, abs(d).^(p-1) .* sign(d)) ;
    end
  end
end

if isempty(dzdy)
  if ~isempty(opts.instanceWeights)
    y1 = bsxfun(@times, opts.instanceWeights, y1) ;
  end
  if opts.aggregate
    y1 = sum(y1(:)) ;
  end
end
if ~isempty(dzdy), y2 = -y1; end
