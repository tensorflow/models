classdef ROIPooling < dagnn.Layer
  % DAGNN.ROIPOOLING  Region of interest pooling layer

  % Copyright (C) 2016 Hakan Bilen.
  % All rights reserved.
  %
  % This file is part of the VLFeat library and is made available under
  % the terms of the BSD license (see the COPYING file).

  properties
    method = 'max'
    subdivisions = [6 6]
    transform = 1
    flatten = false
  end

  methods
    function outputs = forward(obj, inputs, params)
      numROIs = numel(inputs{2}) / 5 ;
      outputs{1} = vl_nnroipool(...
        inputs{1}, inputs{2}, ...
        'subdivisions', obj.subdivisions, ...
        'transform', obj.transform, ...
        'method', obj.method) ;
      if obj.flatten
        outputs{1} = reshape(outputs{1},1,1,[],numROIs) ;
      end
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      numROIs = numel(inputs{2}) / 5 ;
      if obj.flatten
        % unflatten
        derOutputs{1} = reshape(...
          derOutputs{1},obj.subdivisions(1),obj.subdivisions(2),[],numROIs) ;
      end
      derInputs{1} = vl_nnroipool(...
        inputs{1}, inputs{2}, derOutputs{1}, ...
        'subdivisions', obj.subdivisions, ...
        'transform', obj.transform, ...
        'method', obj.method) ;
      derInputs{2} = [];
      derParams = {} ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      if isempty(inputSizes{1})
        n = 0 ;
      else
        n = prod(inputSizes{2})/5 ;
      end
      outputSizes{1} = [obj.subdivisions, inputSizes{1}(3), n] ;
    end

    function obj = ROIPooling(varargin)
      obj.load(varargin) ;
    end
  end
end
