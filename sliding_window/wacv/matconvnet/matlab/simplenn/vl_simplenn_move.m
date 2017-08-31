function net = vl_simplenn_move(net, destination)
%VL_SIMPLENN_MOVE  Move a SimpleNN network between CPU and GPU.
%   NET = VL_SIMPLENN_MOVE(NET, 'gpu') moves the network to the
%   current GPU device. NET = VL_SIMPLENN_MOVE(NET, 'cpu') moves the
%   network to the CPU.
%
%   See also: VL_SIMPLENN().

% Copyright (C) 2014-15 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

switch destination
  case 'gpu', moveop = @(x) gpuArray(x) ;
  case 'cpu', moveop = @(x) gather(x) ;
  otherwise, error('Unknown destination ''%s''.', destination) ;
end
for l=1:numel(net.layers)
  switch net.layers{l}.type
    case {'conv', 'convt', 'bnorm'}
      for f = {'filters', 'biases', 'filtersMomentum', 'biasesMomentum'}
        f = char(f) ;
        if isfield(net.layers{l}, f)
          net.layers{l}.(f) = moveop(net.layers{l}.(f)) ;
        end
      end
      for f = {'weights', 'momentum'}
        f = char(f) ;
        if isfield(net.layers{l}, f)
          for j=1:numel(net.layers{l}.(f))
            net.layers{l}.(f){j} = moveop(net.layers{l}.(f){j}) ;
          end
        end
      end
    otherwise
      % nothing to do ?
  end
end
