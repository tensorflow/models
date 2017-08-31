function move(obj, device)
%MOVE Move the DagNN to either CPU or GPU
%   MOVE(obj, 'cpu') moves the DagNN obj to the CPU.
%
%   MOVE(obj, 'gpu') moves the DagNN obj to the GPU.

% Copyright (C) 2015 Karel Lenc and Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

obj.reset() ;
obj.device = device ;
switch device
  case 'gpu'
    for i=1:numel(obj.params)
      obj.params(i).value = gpuArray(obj.params(i).value) ;
    end
  case 'cpu'
    for i=1:numel(obj.params)
      obj.params(i).value = gather(obj.params(i).value) ;
    end
  otherwise
    error('DEVICE must be either ''cpu'' or ''gpu''.') ;
end
for l = 1:numel(obj.layers)
  obj.layers(l).block.move(device) ;
end
