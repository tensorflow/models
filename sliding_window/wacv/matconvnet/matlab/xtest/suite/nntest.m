classdef nntest < matlab.unittest.TestCase
  properties (ClassSetupParameter)
    device = {'cpu', 'gpu'}
    dataType = {'single', 'double'}
  end

  properties
    currentDevice
    currentDataType
    randn
    rand
    zeros
    ones
    toDevice
    toDataType
    range = 128
  end

  methods (TestClassSetup)
    function generators(test, device, dataType)
      test.currentDevice = device ;
      test.currentDataType = dataType ;
      switch dataType
        case 'single'
          test.toDataType = @(x) single(x) ;
        case 'double'
          test.toDataType = @(x) double(x) ;
      end
      switch device
        case 'gpu'
          gpuDevice ;
          test.randn = @(varargin) test.range * gpuArray.randn(varargin{:},dataType) ;
          test.rand = @(varargin) test.range * gpuArray.rand(varargin{:},dataType) ;
          test.zeros = @(varargin) gpuArray.zeros(varargin{:},dataType) ;
          test.ones = @(varargin) gpuArray.ones(varargin{:},dataType) ;
          test.toDevice = @(x) gpuArray(x) ;
        case 'cpu'
          test.randn = @(varargin) test.range * randn(varargin{:},dataType) ;
          test.rand = @(varargin) test.range * rand(varargin{:},dataType) ;
          test.zeros = @(varargin) zeros(varargin{:},dataType) ;
          test.ones = @(varargin) ones(varargin{:},dataType) ;
          test.toDevice = @(x) gather(x) ;
      end
    end
  end

  methods (TestMethodSetup)
    function seeds(test)
      seed = 0 ;
      switch test.currentDevice
        case 'gpu'
          parallel.gpu.rng(seed, 'combRecursive') ;
        case 'cpu'
          rng(seed, 'combRecursive') ;
      end
    end
  end

  methods
    function der(test, g, x, dzdy, dzdx, delta, tau)
      if nargin < 7
        tau = [] ;
      end
      dzdx_ = test.toDataType(test.numder(g, x, dzdy, delta)) ;
      test.eq(gather(dzdx_), gather(dzdx), tau) ;
    end

    function eq(test,a,b,tau)
      a = gather(a) ;
      b = gather(b) ;
      if nargin > 3 && ~isempty(tau) && tau < 0
        tau_min = -tau ;
        tau = [] ;
      else
        tau_min = 0 ;
      end
      if nargin < 4 || isempty(tau)
        maxv = max([max(a(:)), max(b(:))]) ;
        minv = min([min(a(:)), min(b(:))]) ;
        tau = max(1e-2 * (maxv - minv), 1e-3 * max(maxv, -minv)) ;
      end
      tau = max(tau, tau_min) ;
      if isempty(tau) % can happen if a and b are empty
        tau = 0 ;
      end
      test.verifyThat(b, matlab.unittest.constraints.IsOfClass(class(a))) ;
      tau = feval(class(a), tau) ; % convert to same type as a
      tol = matlab.unittest.constraints.AbsoluteTolerance(tau) ;
      test.verifyThat(a, matlab.unittest.constraints.IsEqualTo(b, 'Within', tol)) ;
    end
  end

  methods (Static)
    function dzdx = numder(g, x, dzdy, delta)
      if nargin < 3
        delta = 1e-3 ;
      end
      dzdy = gather(dzdy) ;
      y = gather(g(x)) ;
      dzdx = zeros(size(x),'double') ;
      for i=1:numel(x)
        x_ = x ;
        x_(i) = x_(i) + delta ;
        y_ = gather(g(x_)) ;
        factors = dzdy .* (y_ - y)/delta ;
        dzdx(i) = dzdx(i) + sum(factors(:)) ;
      end
    end
  end
end
