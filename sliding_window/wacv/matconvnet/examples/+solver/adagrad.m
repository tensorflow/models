function [w, g_sqr] = adagrad(w, g_sqr, grad, opts, lr)
%ADAGRAD
%   Example AdaGrad solver, for use with CNN_TRAIN and CNN_TRAIN_DAG.
%
%   Set the initial learning rate for AdaGrad in the options for
%   CNN_TRAIN and CNN_TRAIN_DAG. Note that a learning rate that works for
%   SGD may be inappropriate for AdaGrad; the default is 0.001.
%
%   If called without any input argument, returns the default options
%   structure.
%
%   Solver options: (opts.train.solverOpts)
%
%   `epsilon`:: 1e-10
%      Small additive constant to regularize variance estimate.
%
%   `rho`:: 1
%      Moving average window for variance update, between 0 and 1 (larger
%      values result in slower/more stable updating). This is similar to
%      RHO in AdaDelta and RMSProp. Standard AdaGrad is obtained with a RHO
%      value of 1 (use total average instead of a moving average).
%
%   A possibly undesirable effect of standard AdaGrad is that the update
%   will monotonically decrease to 0, until training eventually stops. This
%   is because the AdaGrad update is inversely proportional to the total
%   variance of the gradients seen so far.
%   With RHO smaller than 1, a moving average is used instead. This
%   prevents the final update from monotonically decreasing to 0.

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

if nargin == 0  % Return the default solver options
  w = struct('epsilon', 1e-10, 'rho', 1) ;
  return ;
end

g_sqr = g_sqr * opts.rho + grad.^2 ;

w = w - lr * grad ./ (sqrt(g_sqr) + opts.epsilon) ;
