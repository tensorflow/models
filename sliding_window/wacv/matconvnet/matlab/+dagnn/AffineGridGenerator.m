classdef AffineGridGenerator < dagnn.Layer
%DAGNN.AFFINEGRIDGENERATIOR  Generate an affine grid for bilinear resampling
%   This layer maps 1 x 1 x 6 x N affine transforms to 2 x Ho x Wo x N
%   sampling grids compatible with dagnn.BlilinearSampler.

% (c) 2016 Ankush Gupta

 properties
     Ho = 0;
     Wo = 0;
 end

  properties (Transient)
    % the grid (normalized \in [-1,1]) --> this is cached
    % has the size: [HoWo x 2]
    xxyy ;
  end

  methods
    function outputs = forward(obj, inputs, ~)
      % input is a 1x1x6xN TENSOR corresponding to:
      % [ c1 c2 c5 ]
      % [ c3 c4 c6 ]
      % [  0  0  1 ]
      % i.e., [d1] = [c1 c2]  * [d1] + [c5]
      %       [d2]   [c3 c4]    [d2]   [c6]
      % where, di is the i-th dimension.
      % 
      % OUTPUT is a 2xHoxWoxN grid which corresponds to applying
      % the above affine transform to the [-1,1] normalized x,y
      % coordinates.

      %fprintf('affineGridGenerator forward\n');
      useGPU = isa(inputs{1}, 'gpuArray');

      % reshape the tfm params into matrices:
      A = inputs{1};
      nbatch = size(A,4);
      A = reshape(A, 2,3,nbatch);
      L = A(:,1:2,:);
      L = reshape(L,2,2*nbatch); % linear part

      % generate the grid coordinates:
      if isempty(obj.xxyy)
        obj.initGrid(useGPU);
      end

      % transform the grid:
      t = A(:,3,:); % translation
      t = reshape(t,1,2*nbatch);
      g = bsxfun(@plus, obj.xxyy * L, t); % apply the transform
      g = reshape(g, obj.Wo,obj.Ho,2,nbatch);

      % cudnn compatibility:
      g = permute(g, [3,2,1,4]);

      outputs = {g};
    end

    function [derInputs, derParams] = backward(obj, inputs, ~, derOutputs)

      useGPU = isa(derOutputs{1}, 'gpuArray');
      dY = derOutputs{1};
      nbatch = size(dY,4);

      % cudnn compatibility:
      dY = permute(dY, [3,2,1,4]);

      % create the gradient buffer:
      dA = zeros([2,3,nbatch], 'single');
      if useGPU, dA = gpuArray(dA); end

      dY = reshape(dY, obj.Ho*obj.Wo, 2*nbatch);
      % gradient wrt the linear part:
      dL = obj.xxyy' * dY;
      dL = reshape(dL,2,2,nbatch);
      dA(:,1:2,:) = dL;

      % gradient wrt translation (or bias):
      dt = reshape(sum(dY,1),2,1,nbatch);
      dA(:,3,:) = dt;

      dA = reshape(dA, size(inputs{1}));
      derInputs = {dA};
      derParams = {};
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      nBatch = inputSizes{1}(4);
      outputSizes = {[2, obj.Ho, obj.Wo, nBatch]};
    end

    function obj = AffineGridGenerator(varargin)
      obj.load(varargin) ;
      % get the output sizes:
      obj.Ho = obj.Ho ;
      obj.Wo = obj.Wo ;
      obj.xxyy = [] ;
    end

    function obj = reset(obj)
      reset@dagnn.Layer(obj) ;
      obj.xxyy = [] ;
    end

    function initGrid(obj, useGPU)
      % initialize the grid:
      % this is a constant
      xi = linspace(-1, 1, obj.Ho);
      yi = linspace(-1, 1, obj.Wo);

      [yy,xx] = meshgrid(xi,yi);
      xxyy = [yy(:), xx(:)] ; % Mx2
      if useGPU
        xxyy = gpuArray(xxyy);
      end
      obj.xxyy = xxyy ;
    end

  end
end
