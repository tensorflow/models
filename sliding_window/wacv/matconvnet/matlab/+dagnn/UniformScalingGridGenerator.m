%DAGNN.UNIFORMSCALINGGRIDGENERATOR  Generate an iso-tropic scaling + translation
%   grid for bilinear resampling.
%   This layer maps 1 x 1 x 3 x N transformation parameters corresponding to
%   the scale and translation y and x respectively, to 2 x Ho x Wo x N
%   sampling grids compatible with dagnn.BlilinearSampler.

% (c) 2016 Ankush Gupta
classdef UniformScalingGridGenerator < dagnn.Layer

 properties
     Ho = 0;
     Wo = 0;
 end

  properties (Transient)
    % the grid --> this is cached
    % has the size: [2 x HoWo]
    xxyy ;
  end

  methods

    function outputs = forward(obj, inputs, ~)
      % input is a 1x1x3xN TENSOR corresponding to:
      % [  s 0 ty ]
      % [  0 s tx ]
      %
      % OUTPUT is a 2xHoxWoxN grid

      % reshape the tfm params into matrices:
      T = inputs{1};
      % check shape:
      sz_T = size(T);
      assert(all(sz_T(1:3) == [1 1 3]), 'transforms have incorrect shape');
      nbatch = size(T,4);
      S = reshape(T(1,1,1,:), 1,1,nbatch); % x,y scaling
      t = reshape(T(1,1,2:3,:), 2,1,nbatch); % translation
      % generate the grid coordinates:
      useGPU = isa(T, 'gpuArray');
      if isempty(obj.xxyy)
        obj.initGrid(useGPU);
      end
      % transform the grid:
      g = bsxfun(@times, obj.xxyy, S); % scale
      g = bsxfun(@plus, g, t); % translate
      g = reshape(g, 2,obj.Ho,obj.Wo,nbatch);
      outputs = {g};
    end

    function [derInputs, derParams] = backward(obj, inputs, ~, derOutputs)
      dY = derOutputs{1};
      useGPU = isa(dY, 'gpuArray');
      nbatch = size(dY,4);

      % create the gradient buffer:
      dA = zeros([1,1,3,nbatch], 'single');
      if useGPU, dA = gpuArray(dA); end

      dY  = reshape(dY, 2,obj.Ho*obj.Wo,nbatch);
      % gradient wrt the linear part:
      dA(1,1,1,:) = reshape(obj.xxyy,1,[]) * reshape(dY, [],nbatch);
      % gradient wrt translation (or bias):
      dA(1,1,2:3,:) = reshape(sum(dY,2),1,1,2,[]);

      derInputs = {dA};
      derParams = {};
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      nBatch = inputSizes{1}(4);
      outputSizes = {[2, obj.Ho, obj.Wo, nBatch]};
    end

    function obj = UniformScalingGridGenerator(varargin)
      obj.load(varargin);
      % get the output sizes:
      obj.Ho = obj.Ho;
      obj.Wo = obj.Wo;
      obj.xxyy = [];
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
      [yy,xx] = meshgrid(yi,xi);
      xxyy = [xx(:), yy(:)]' ; % 2xM
      if useGPU
        xxyy = gpuArray(xxyy);
      end
      obj.xxyy = xxyy ;
    end

  end
end
