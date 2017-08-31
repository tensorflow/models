classdef ParameterServer < handle
  properties
    params
    paramsRegister
    method
    prefix
    memoryMapFile
    memoryMap
    pinnedMemory
    otherLabs
    inplace
    tmoveOpts
  end

  methods
    function obj = ParameterServer(varargin)
      obj.prefix = 'mcn' ;
      obj.params = struct('name', {}, 'shape', {}, ...
                          'dataType', {}, 'deviceType', {}, ...
                          'value', {}) ;
      obj.paramsRegister = struct() ;
      switch computer('arch')
        case 'win64'
          obj.method = 'mmap' ;
        otherwise
          obj.method = 'tmove' ;
      end
      obj.pinnedMemory = true ;
      obj.inplace = true ;
      obj.tmoveOpts = { } ;
      obj = vl_argparse(obj, varargin) ;

      % Compute the memory map file from the prefix.  This is used only by
      % the 'mmap' mode.
      obj.memoryMapFile = fullfile(tempdir, sprintf('%s.bin',obj.prefix)) ;
    end

    function register(obj, name, shape, dataType, deviceType)
      x = zeros(shape, dataType) ;
      if strcmp(deviceType,'gpu')
        x = gpuArray(x) ;
      end
      obj.params(end+1).name = name ;
      obj.params(end).shape = shape ;
      obj.params(end).dataType = dataType ;
      obj.params(end).deviceType = deviceType ;
      obj.params(end).isGPU = strcmp(deviceType, 'gpu') ;
      obj.params(end).value =  x ;
      obj.paramsRegister.(name) = numel(obj.params) ;
    end

    function delete(obj)
      obj.stop()
    end

    function start(obj)
      switch obj.method
        case 'mmap'
          if labindex == 1 && exist(obj.memoryMapFile)
            delete(obj.memoryMapFile) ;
          end
          labBarrier() ;
          obj = startWithMMap(obj) ;
        case 'tmove'
          obj = startWithTMove(obj) ;
      end
    end

    function stop(obj)
      switch obj.method
        case 'mmap', stopWithMMap(obj) ;
        case 'tmove', stopWithTMove(obj) ;
      end
    end

    function push(obj, name, value)
      p = obj.paramsRegister.(name) ;
      obj.pushWithIndex(p, value) ;
    end

    function value = pull(obj, name)
      p = obj.paramsRegister.(name) ;
      value = obj.pullWithIndex(p) ;
    end

    function pushWithIndex(obj, p, value)
      switch obj.method
        case 'mmap'
          obj.params(p).value = value ;

        case 'tmove'
          if obj.params(p).isGPU && obj.inplace
            obj.params(p).value = value ;
            vl_tmove('push', obj.params(p).name, value, 'inplace', obj.tmoveOpts{:}) ;
          else
            vl_tmove('push', obj.params(p).name, value, obj.tmoveOpts{:}) ;
          end

        otherwise
          assert(false) ;
      end
    end

    function value = pullWithIndex(obj,p)
      switch obj.method
        case 'mmap'
          for l = obj.otherLabs
            obj.params(p).value = obj.params(p).value ...
                + obj.memoryMap.Data(l).(obj.params(p).name) ;
          end
          value = obj.params(p).value ;

        case 'tmove'
          if obj.params(p).isGPU && obj.inplace
            vl_tmove('pull', obj.params(p).name, 'inplace', obj.tmoveOpts{:}) ;
            value = obj.params(p).value ;
          else
            value = vl_tmove('pull', obj.params(p).name, obj.tmoveOpts{:}) ;
          end
      end
    end

    function sync(obj)
      switch obj.method
        case 'mmap'
          for p = 1:numel(obj.params)
            if ~obj.pinnedMemory || ~obj.params(p).isGPU
              obj.memoryMap.Data(labindex).(obj.params(p).name) = ...
                  gather(obj.params(p).value) ;
            else
              vl_cudatool('cudaCopyDeviceToHost', ...
                       obj.memoryMap.Data(labindex).(obj.params(p).name), ...
                       obj.params(p).value) ;
            end
          end
          labBarrier() ;
        case 'tmove'
          % nothing to do
      end
    end
  end

  methods (Access = protected)
    function obj = startWithMMap(obj)
      obj.otherLabs = setdiff(1:numlabs, labindex) ;
      format = {} ;
      for i=1:numel(obj.params)
        format(i,1:3) = {obj.params(i).dataType, ...
                         obj.params(i).shape, ...
                         obj.params(i).name} ;
      end
      if labindex == 1
        f = fopen(obj.memoryMapFile,'wb') ;
        for g = 1:numlabs
          for i = 1:size(format,1)
            fwrite(f,zeros(format{i,2},format{i,1}),format{i,1}) ;
          end
        end
        fclose(f) ;
      end
      labBarrier() ;
      obj.memoryMap = memmapfile(obj.memoryMapFile, ...
                                 'Format', format, ...
                                 'Repeat', numlabs, ...
                                 'Writable', true) ;
      if obj.pinnedMemory && any([obj.params.isGPU])
        for i=1:numel(obj.memoryMap.Data)
          names = fieldnames(obj.memoryMap.Data(i))' ;
          for name = names
            vl_cudatool('cudaRegister', obj.memoryMap.Data(i).(char(name))) ;
          end
        end
      end
    end

    function stopWithMMap(obj)
      if ~isempty(obj.memoryMap)
        if obj.pinnedMemory && any([obj.params.isGPU])
          for i=1:numel(obj.memoryMap.Data)
            names = fieldnames(obj.memoryMap.Data(i))' ;
            for name = names
              vl_cudatool('cudaUnregister', obj.memoryMap.Data(i).(char(name))) ;
            end
          end
        end
        obj.memoryMap = [] ;
        if labindex == 1
          delete(obj.memoryMapFile) ;
        end
      end
    end

    function obj = startWithTMove(obj)
      format = {} ;
      for i=1:numel(obj.params)
        format(i,1:4) = {obj.params(i).dataType, ...
                         obj.params(i).shape, ...
                         obj.params(i).name, ...
                         obj.params(i).deviceType} ;
      end
      vl_tmove('reset', obj.tmoveOpts{:}) ;
      vl_tmove('init', format, ...
               labindex, numlabs, ...
               'prefix', obj.prefix, ...
               obj.tmoveOpts{:}) ;
    end

    function stopWithTMove(obj)
      vl_tmove('reset', obj.tmoveOpts{:}) ;
    end

  end
end