% Wrapper for BilinearSampler block:
% (c) 2016 Ankush Gupta

classdef BilinearSampler < dagnn.Layer
  methods
    function outputs = forward(obj, inputs, params)
      outputs = vl_nnbilinearsampler(inputs{1}, inputs{2});
      outputs = {outputs};
    end

    function [derInputs, derParams] = backward(obj, inputs, param, derOutputs)
      [dX,dG] = vl_nnbilinearsampler(inputs{1}, inputs{2}, derOutputs{1});
      derInputs = {dX,dG};
      derParams = {};
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      xSize = inputSizes{1};
      gSize = inputSizes{2};
      outputSizes = {[gSize(2), gSize(3), xSize(3), xSize(4)]};
    end

    function obj = BilinearSampler(varargin)
      obj.load(varargin);
    end
  end
end
