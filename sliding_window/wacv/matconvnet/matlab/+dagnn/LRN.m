classdef LRN < dagnn.ElementWise
  properties
    param = [5 1 0.0001/5 0.75]
  end

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nnnormalize(inputs{1}, obj.param) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, param, derOutputs)
      derInputs{1} = vl_nnnormalize(inputs{1}, obj.param, derOutputs{1}) ;
      derParams = {} ;
    end

    function obj = LRN(varargin)
      obj.load(varargin) ;
    end
  end
end
