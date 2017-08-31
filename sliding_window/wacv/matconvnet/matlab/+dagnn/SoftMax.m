classdef SoftMax < dagnn.ElementWise
  methods
    function outputs = forward(self, inputs, params)
      outputs{1} = vl_nnsoftmax(inputs{1}) ;
    end

    function [derInputs, derParams] = backward(self, inputs, params, derOutputs)
      derInputs{1} = vl_nnsoftmax(inputs{1}, derOutputs{1}) ;
      derParams = {} ;
    end

    function obj = SoftMax(varargin)
      obj.load(varargin) ;
    end
  end
end
