classdef NormOffset < dagnn.ElementWise
  properties
    param = [1 0.5]
  end

  methods
    function outputs = forward(obj, inputs, params)
      outputs{1} = vl_nnnoffset(inputs{1}, obj.param) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, param, derOutputs)
      derInputs{1} = vl_nnnoffset(inputs{1}, obj.param, derOutputs{1}) ;
      derParams = {} ;
    end

    function obj = NormOffset(varargin)
      obj.load(varargin) ;
    end
  end
end
