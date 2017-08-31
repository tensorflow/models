classdef SpatialNorm < dagnn.ElementWise
  properties
    param = [2 2 10 2]
  end

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nnspnorm(inputs{1}, obj.param) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, param, derOutputs)
      derInputs{1} = vl_nnspnorm(inputs{1}, obj.param, derOutputs{1}) ;
      derParams = {} ;
    end

    function obj = SpatialNorm(varargin)
      obj.load(varargin) ;
    end
  end
end
