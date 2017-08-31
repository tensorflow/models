classdef DropOut < dagnn.ElementWise
  properties
    rate = 0.5
    frozen = false
  end

  properties (Transient)
    mask
  end

  methods
    function outputs = forward(obj, inputs, params)
      if strcmp(obj.net.mode, 'test')
        outputs = inputs ;
        return ;
      end
      if obj.frozen & ~isempty(obj.mask)
        outputs{1} = vl_nndropout(inputs{1}, 'mask', obj.mask) ;
      else
        [outputs{1}, obj.mask] = vl_nndropout(inputs{1}, 'rate', obj.rate) ;
      end
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      if strcmp(obj.net.mode, 'test')
        derInputs = derOutputs ;
        derParams = {} ;
        return ;
      end
      derInputs{1} = vl_nndropout(inputs{1}, derOutputs{1}, 'mask', obj.mask) ;
      derParams = {} ;
    end

    % ---------------------------------------------------------------------
    function obj = DropOut(varargin)
      obj.load(varargin{:}) ;
    end

    function obj = reset(obj)
      reset@dagnn.ElementWise(obj) ;
      obj.mask = [] ;
      obj.frozen = false ;
    end
  end
end
