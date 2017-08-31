function [w, state] = adadelta(w, state, grad, opts, ~)
%ADADELTA
%   Example AdaDelta solver, for use with CNN_TRAIN and CNN_TRAIN_DAG.
%
%   AdaDelta sets its own learning rate, so any learning rate set in the
%   options of CNN_TRAIN and CNN_TRAIN_DAG will be ignored.
%
%   If called without any input argument, returns the default options
%   structure.
%
%   Solver options: (opts.train.solverOpts)
%
%   `epsilon`:: 1e-6
%      Small additive constant to regularize variance estimate.
%
%   `rho`:: 0.9
%      Moving average window for variance update, between 0 and 1 (larger
%      values result in slower/more stable updating).

% Copyright (C) 2016 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

if nargin == 0  % Return the default solver options
  w = struct('epsilon', 1e-6, 'rho', 0.9) ;
  return ;
end

if isequal(state, 0)  % First iteration, initialize state struct
  state = struct('g_sqr', 0, 'delta_sqr', 0) ;
end

rho = opts.rho ;

state.g_sqr = state.g_sqr * rho + grad.^2 * (1 - rho) ;
new_delta = -sqrt((state.delta_sqr + opts.epsilon) ./ ...
                  (state.g_sqr + opts.epsilon)) .* grad ;
state.delta_sqr = state.delta_sqr * rho + new_delta.^2 * (1 - rho) ;

w = w + new_delta ;
