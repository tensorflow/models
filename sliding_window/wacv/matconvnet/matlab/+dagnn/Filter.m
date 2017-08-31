classdef Filter < dagnn.Layer
  properties
    pad = [0 0 0 0]
    stride = [1 1]
    dilate = [1 1]
  end
  methods
    function set.pad(obj, pad)
      if numel(pad) == 1
        obj.pad = [pad pad pad pad] ;
      elseif numel(pad) == 2
        obj.pad = pad([1 1 2 2]) ;
      else
        obj.pad = pad ;
      end
    end

    function set.stride(obj, stride)
      if numel(stride) == 1
        obj.stride = [stride stride] ;
      else
        obj.stride = stride ;
      end
    end

    function set.dilate(obj, dilate)
      if numel(dilate) == 1
        obj.dilate = [dilate dilate] ;
      else
        obj.dilate = dilate ;
      end
    end

    function kernelSize = getKernelSize(obj)
      kernelSize = [1 1] ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      ks = obj.getKernelSize() ;
      ke = (ks - 1) .* obj.dilate + 1 ;
      outputSizes{1} = [...
        fix((inputSizes{1}(1) + obj.pad(1) + obj.pad(2) - ke(1)) / obj.stride(1)) + 1, ...
        fix((inputSizes{1}(2) + obj.pad(3) + obj.pad(4) - ke(2)) / obj.stride(2)) + 1, ...
        1, ...
        inputSizes{1}(4)] ;
    end

    function rfs = getReceptiveFields(obj)
      ks = obj.getKernelSize() ;
      ke = (ks - 1) .* obj.dilate + 1 ;
      y1 = 1 - obj.pad(1) ;
      y2 = 1 - obj.pad(1) + ke(1) - 1 ;
      x1 = 1 - obj.pad(3) ;
      x2 = 1 - obj.pad(3) + ke(2) - 1 ;
      h = y2 - y1 + 1 ;
      w = x2 - x1 + 1 ;
      rfs.size = [h, w] ;
      rfs.stride = obj.stride ;
      rfs.offset = [y1+y2, x1+x2]/2 ;
    end
  end
end
