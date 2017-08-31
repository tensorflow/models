classdef nnbilinearsampler < nntest
  properties (TestParameter)
    % input/output sizes:
    ih = {8 16}
    iw = {4 8 16}
    oh = {8}
    ow = {4 8}
    method_type = {'cuda', 'cudnn'}
    multiple_grids = {true false}
  end

  methods (Test)

    function check_der_x(test, ih, iw, oh, ow, multiple_grids)
      if ~strcmp(test.currentDevice ,'cpu'), return; end
      [ims,grids,dout] = test.init_params(ih,iw,oh,ow,4,multiple_grids); % param-init
      [dx,~] = vl_nnbilinearsampler(ims,grids,dout); % analytic deriv
      test.der(@(inp)vl_nnbilinearsampler(inp,grids), ims, dout, dx, 1e-2); % numcheck
    end
% 
%     function check_der_grid(test, ih, iw, oh, ow, multiple_grids)
%       if ~strcmp(test.currentDevice ,'cpu'), return; end
%       [ims,grids,dout] = test.init_params(ih,iw,oh,ow,4,multiple_grids); % param-init
%       [~,dgrid] = vl_nnbilinearsampler(ims,grids,dout); % analytic deriv
%       test.der(@(inp)vl_nnbilinearsampler(ims,inp), grids, dout, dgrid, 1e-2); % numcheck
%     end
    
    % function check_der_x_gpu(test, ih, iw, oh, ow, method_type, multiple_grids)
    %   if ~strcmp(test.currentDevice ,'gpu'), return; end
    %   opts = {};
    %   switch method_type
    %     case 'cuda'
    %       opts = {'NoCudnn'};
    %     case 'cudnn'
    %       opts = {'Cudnn'};
    %   end
    %   [ims,grids,dout] = test.init_params(ih,iw,oh,ow,4,multiple_grids); % param-init
    %   [dx,~] = vl_nnbilinearsampler(ims,grids,dout,opts{:}); % analytic deriv
    %   test.der(@(inp)vl_nnbilinearsampler(inp,grids,opts{:}), ims, dout, dx, 1e-2); % numcheck
    % end

    % function check_der_grid_gpu(test, ih, iw, oh, ow, method_type, multiple_grids)
    %   if ~strcmp(test.currentDevice ,'gpu'), return; end
    %   opts = {};
    %   switch method_type
    %     case 'cuda'
    %       opts = {'NoCudnn'};
    %     case 'cudnn'
    %       opts = {'Cudnn'};
    %   end
    %   [ims,grids,dout] = test.init_params(ih,iw,oh,ow,4,multiple_grids); % param-init
    %   [~,dgrid] = vl_nnbilinearsampler(ims,grids,dout,opts{:}); % analytic deriv
    %   test.der(@(inp)vl_nnbilinearsampler(ims,inp,opts{:}), grids, dout, dgrid, 1e-2); % numcheck
    % end

    function fwd_consistency(test, ih, iw, oh, ow, multiple_grids)
      if ~strcmp(test.currentDevice ,'gpu'), return; end

      [ims,grids,~] = test.init_params(ih,iw,oh,ow,4,multiple_grids); % param-init
      %out_cpu = vl_nnbilinearsampler(gather(ims),gather(grids));
      out_cuda = vl_nnbilinearsampler(ims,grids,'NoCudnn');
      out_cudnn = vl_nnbilinearsampler(ims,grids,'Cudnn');
      out_cuda = gather(out_cuda);
      out_cudnn = gather(out_cudnn);
      % compare:
      %test.eq(out_cpu,  out_cudnn);
      test.eq(out_cuda, out_cudnn);
      %test.eq(out_cuda, out_cpu);
    end

    function bwd_grid_consistency(test, ih, iw, oh, ow, multiple_grids)
      if ~strcmp(test.currentDevice ,'gpu'), return; end

      [ims,grids,dout] = test.init_params(ih,iw,oh,ow,4,multiple_grids); % param-init
      %[~,dgrid_cpu] = vl_nnbilinearsampler(gather(ims),gather(grids),gather(dout));
      [~,dgrid_cuda] = vl_nnbilinearsampler(ims,grids,dout,'NoCudnn');
      [~,dgrid_cudnn] = vl_nnbilinearsampler(ims,grids,dout,'Cudnn');
      dgrid_cuda = gather(dgrid_cuda);
      dgrid_cudnn = gather(dgrid_cudnn);
      % compare:
      %test.eq(dgrid_cpu, dgrid_cudnn);
      test.eq(dgrid_cuda,dgrid_cudnn);
      %test.eq(dgrid_cuda,dgrid_cpu);
    end

    function bwd_data_consistency(test, ih, iw, oh, ow, multiple_grids)
      if ~strcmp(test.currentDevice ,'gpu'), return; end

      [ims,grids,dout] = test.init_params(ih,iw,oh,ow,4,multiple_grids); % param-init
      %[dData_cpu,~] = vl_nnbilinearsampler(gather(ims),gather(grids),gather(dout));
      [dData_cuda,~] = vl_nnbilinearsampler(ims,grids,dout,'NoCudnn');
      [dData_cudnn,~] = vl_nnbilinearsampler(ims,grids,dout,'Cudnn');
      dData_cuda = gather(dData_cuda);
      dData_cudnn = gather(dData_cudnn);
      % compare:
      %test.eq(dData_cpu,  dData_cudnn);
      test.eq(dData_cuda, dData_cudnn);
      %test.eq(dData_cuda, dData_cpu);
    end

  end

  methods 
    function [x,grid,dout] = init_params(test, ih,iw,oh,ow,n, multiple_grids)
      % initialize a batch of images:
      assert(mod(n,2)==0, 'n should be a multiple of 2.');

      i1 = imread('peppers.png');
      i2 = imread('pears.png');
      i1 = imresize(i1,[ih,iw]);
      i2 = imresize(i2,[ih,iw]); 

      nin = n;
      if multiple_grids, nin = n/2; end
      x = zeros([ih,iw,3,nin], test.currentDataType);
      for i = 1:(nin/2)
        x(:,:,:,1 + 2*(i-1)) = i1(:,:,:);
        x(:,:,:,2 + 2*(i-1)) = i2(:,:,:);
      end
      x = x / 255;

      % initialize grids:
      tf1 = [1 0 1;
             0 1  0];
      tf2 = [1 0 0;
             0 1 1];
      tf3 = [1/2  0  0;
              0  1/2 0];
      th = -pi/4;
      tf4 = [cos(th)     -sin(th)  0;
             sin(th)      cos(th)  0];
      tf4(:,1:2) = tf4(:,1:2)'; % take the transpose of the rotation
      tf = {tf1,tf2,tf3,tf4};
      aff = zeros(1,1,6,n, test.currentDataType);
      for i=1:n
        aff(1,1,:,i) = tf{i}(:);
      end
      gridGen = dagnn.AffineGridGenerator('Ho',oh,'Wo',ow);
      grid = gridGen.forward({aff},{});
      grid = grid{1};

      % get fake derivative:
      dout = test.randn(oh,ow,3,n);
      % [optionally] move to gpu:
      x = test.toDevice(x);
      grid = test.toDevice(grid);
    end
  end

end % class
