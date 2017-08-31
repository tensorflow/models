function [w, state] = adam(w, state, grad, opts, lr)
%ADAM
%   Adam solver for use with CNN_TRAIN and CNN_TRAIN_DAG
%
%   See [Kingma et. al., 2014](http://arxiv.org/abs/1412.6980)
%    |  ([pdf](http://arxiv.org/pdf/1412.6980.pdf)).
%
%   If called without any input argument, returns the default options
%   structure. Otherwise provide all input arguments.
%   
%   W is the vector/matrix/tensor of parameters. It can be single/double
%   precision and can be a `gpuArray`.
%
%   STATE is as defined below and so are supported OPTS.
%
%   GRAD is the gradient of the objective w.r.t W
%
%   LR is the learning rate, referred to as \alpha by Algorithm 1 in 
%   [Kingma et. al., 2014].
%
%   Solver options: (opts.train.solverOpts)
%
%   `beta1`:: 0.9
%      Decay for 1st moment vector. See algorithm 1 in [Kingma et.al. 2014]
%
%   `beta2`:: 0.999
%      Decay for 2nd moment vector
%
%   `eps`:: 1e-8
%      Additive offset when dividing by state.v
%
%   The state is initialized as 0 (number) to start with. The first call to
%   this function will initialize it with the default state consisting of
%
%   `m`:: 0
%      First moment vector
%
%   `v`:: 0
%      Second moment vector
%
%   `t`:: 0
%      Global iteration number across epochs
%
%   This implementation borrowed from torch optim.adam

% Copyright (C) 2016 Aravindh Mahendran.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

if nargin == 0 % Returns the default solver options
  w = struct('beta1', 0.9, 'beta2', 0.999, 'eps', 1e-8) ;
  return ;
end

if isequal(state, 0) % start off with state = 0 so as to get default state
  state = struct('m', 0, 'v', 0, 't', 0);
end

% update first moment vector `m`
state.m = opts.beta1 * state.m + (1 - opts.beta1) * grad ;

% update second moment vector `v`
state.v = opts.beta2 * state.v + (1 - opts.beta2) * grad.^2 ;

% update the time step
state.t = state.t + 1 ;

% This implicitly corrects for biased estimates of first and second moment
% vectors
lr_t = lr * (((1 - opts.beta2^state.t)^0.5) / (1 - opts.beta1^state.t)) ;

% Update `w`
w = w - lr_t * state.m ./ (state.v.^0.5 + opts.eps) ;
