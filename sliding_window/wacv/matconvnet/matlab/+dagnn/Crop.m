classdef Crop < dagnn.ElementWise
%CROP DagNN cropping layer.
%    This is a pecurial layer from FCN. It crops inputs{1} to
%    match the size of inputs{2} (starting with a base crop amount).
%    A future version

  properties
    crop = [0 0]
  end

  properties (Transient)
    inputSizes = {}
  end

  methods
    function crop = getAdaptedCrops(obj)
      cropv = obj.inputSizes{1}(1) - obj.inputSizes{2}(1) ;
      cropu = obj.inputSizes{1}(2) - obj.inputSizes{2}(2) ;
      cropv1 = max(0, cropv - obj.crop(1)) ;
      cropu1 = max(0, cropu - obj.crop(2)) ;
      crop = [cropv - cropv1, cropv1, cropu - cropu1, cropu1] ;
    end

    function outputs = forward(obj, inputs, params)
      obj.inputSizes = cellfun(@size, inputs, 'UniformOutput', false) ;
      adjCrop = obj.getAdaptedCrops() ;
      outputs{1} = vl_nncrop(inputs{1}, adjCrop) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      adjCrop = obj.getAdaptedCrops() ;
      derInputs{1} = vl_nncrop(inputs{1}, adjCrop, derOutputs{1}, obj.inputSizes{1}) ;
      derInputs{2} = [] ;
      derParams = {} ;
    end

    function reset(obj)
      obj.inputSizes = {} ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      obj.inputSizes = inputSizes ;
      crop = obj.getAdaptedCrops() ;
      outputSizes{1} = inputSizes{1} - [crop(1)+crop(2), crop(3)+crop(4), 0, 0] ;
    end

    function rfs = getReceptiveFields(obj)
      rfs(1,1).size = [1 1] ;
      rfs(1,1).stride = [1 1] ;
      rfs(1,1).offset = 1 + obj.crop ;
      rfs(2,1).size = [] ;
      rfs(2,1).stride = [] ;
      rfs(2,1).offset = [] ;
    end

    function obj = Crop(varargin)
      obj.load(varargin) ;
    end
  end
end
