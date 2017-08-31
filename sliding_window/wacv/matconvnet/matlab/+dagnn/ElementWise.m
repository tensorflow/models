classdef ElementWise < dagnn.Layer
%ELEMENTWISE DagNN layers that operate at individual spatial locations
  methods
    function [outputSizes, transforms] = forwardGeometry(self, inputSizes, paramSizes)
      outputSizes = inputSizes ;
      transforms = {eye(6)} ;
    end

    function rfs = getReceptiveFields(obj)
      rfs.size = [1 1] ;
      rfs.stride = [1 1] ;
      rfs.offset = [1 1] ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes = inputSizes ;
    end
  end
end
