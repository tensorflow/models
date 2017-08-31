classdef nnconvt < nntest
  properties (TestParameter)
    depth = {1 2 3}
    numImages = {1 2 3 4}
    numFilters = {1 2 3}
    upx = {1 2 3}
    upy = {1 2 3}
    padx1 = {1 2 3}
    padx2 = {1 2 3}
    pady1 = {1 2 3}
    pady2 = {1 2 3}
    up = {1 2}
    fsx = {1 2}
    crop = {1 2 3 4 5 6 7 8}
    numGroups = {1 2 3}
  end

  methods (Test)
    function basic(test, depth, numImages, numFilters)
      m = depth ;
      n = numImages ;
      k = numFilters;
      x = test.randn(10,12,m,n) ;
      f = test.randn(3,4,k,m) ;
      b = test.randn(1,k) ;
      y = vl_nnconvt(x,f,b) ;
      dzdy = test.randn(size(y)) ;
      [dzdx,dzdf,dzdb] = vl_nnconvt(x,f,b,dzdy) ;
      test.der(@(x) vl_nnconvt(x,f,b), x, dzdy, dzdx, test.range * 1e-2) ;
      test.der(@(f) vl_nnconvt(x,f,b), f, dzdy, dzdf, test.range * 1e-2) ;
      test.der(@(b) vl_nnconvt(x,f,b), b, dzdy, dzdb, test.range) ;
    end

    function upsample_crop(test,upx,upy,padx1,pady1,padx2,pady2)
      m = 3 ; n = 2 ; k = 3;
      opts = {'upsample',[upy upx],'crop',[pady1 pady2 padx1 padx2]} ;
      x = test.randn(5,6,m,n) ;
      f = test.randn(3,4,k,m) ;
      b = test.randn(1,k) ;
      y = vl_nnconvt(x,f,b,opts{:}) ;
      dzdy = test.randn(size(y)) ;
      [dzdx,dzdf,dzdb] = vl_nnconvt(x,f,b,dzdy,opts{:}) ;
      test.der(@(x) vl_nnconvt(x,f,b,opts{:}), x, dzdy, dzdx, test.range * 1e-2) ;
      test.der(@(f) vl_nnconvt(x,f,b,opts{:}), f, dzdy, dzdf, test.range * 1e-2) ;
      test.der(@(b) vl_nnconvt(x,f,b,opts{:}), b, dzdy, dzdb, test.range) ;
    end

    function grouped_filters(test, numGroups, depth, numFilters)
      ng = numGroups ;
      m = depth ;
      k = numFilters ;
      n = 3 ;
      opts = {'numgroups',ng} ;
      x = test.randn(10,12,m*ng,n) ;
      f = test.randn(3,4,k,m*ng) ;
      b = test.randn(1,k*ng) ;
      y = vl_nnconvt(x,f,b,opts{:}) ;
      dzdy = test.randn(size(y)) ;
      [dzdx,dzdf,dzdb] = vl_nnconvt(x,f,b,dzdy,opts{:}) ;
      test.der(@(x) vl_nnconvt(x,f,b,opts{:}), x, dzdy, dzdx, test.range * 1e-2) ;
      test.der(@(f) vl_nnconvt(x,f,b,opts{:}), f, dzdy, dzdf, test.range * 1e-2) ;
      test.der(@(b) vl_nnconvt(x,f,b,opts{:}), b, dzdy, dzdb, test.range) ;
    end

    function one_one_image(test,up,fsx,crop)
      fsx = fsx*up ;
      if crop > fsx-1, return ; end
      m = 3 ;
      n = 4 ;
      k = 3 ;
      fsy = fsx * 3 ;
      x = test.randn(1,1,m,n) ;
      f = test.randn(fsy,fsx,k,m) ;
      b = test.randn(1,k) ;
      croph = floor(crop/2) ;
      opts = {'crop', [croph, crop-croph, croph, crop-croph], 'upsample', [up up]} ;
      y = vl_nnconvt(x,f,b,opts{:}) ;
      dzdy = test.randn(size(y)) ;
      [dzdx,dzdf,dzdb] = vl_nnconvt(x,f,b,dzdy,opts{:}) ;
      test.der(@(x) vl_nnconvt(x,f,b,opts{:}), x, dzdy, dzdx, test.range * 1e-2) ;
      test.der(@(f) vl_nnconvt(x,f,b,opts{:}), f, dzdy, dzdf, test.range * 1e-2) ;
      test.der(@(b) vl_nnconvt(x,f,b,opts{:}), b, dzdy, dzdb, test.range * 1e-1) ;
    end

    function test_gpu_correctnes(test)
      if ~strcmp(test.currentDevice, 'gpu'), return ; end
      opts = {...
        {'crop', [0 0 0 0], 'upsample', [1 1]}, ...
        {'crop', [5 5 8 8], 'upsample', [1 1]}, ...
        {'crop', [5 5 8 8], 'upsample', [3 2]}} ;

      variants = {{'nocudnn'}, ...
                  {'cudnn', 'cudnnworkspacelimit', 0}, ...
                  {'cudnn', 'cudnnworkspacelimit', +inf}} ;

      fh = 11 ;
      fw = 11 ;
      fn = 10 ;
      n = 32 ;
      depth = 32 ;
      x = test.randn(32,32,depth,n) ;
      w = test.randn(fh,fw,fn,depth) ;
      b = test.randn(1,fn) ;

      for o = 1:numel(opts)
        for v = 1:numel(variants)
          %args = horzcat(variants{v}, opts{o}, {'verbose'}) ;
          args = horzcat(variants{v}, opts{o}) ;
          y = vl_nnconvt(x,w,b,args{:}) ;
          dzdy = test.randn(size(y)) ;
          [dzdx,dzdw,dzdb] = vl_nnconvt(x,w,b,dzdy,args{:}) ;

          dzdy_ = gather(dzdy) ;
          y_ = vl_nnconvt(gather(x), gather(w), gather(b), opts{o}{:}) ;
          [dzdx_,dzdw_,dzdb_] = vl_nnconvt(gather(x),gather(w),gather(b), ...
                                           gather(dzdy), opts{o}{:}) ;

          test.eq(y, y_) ;
          test.eq(dzdx, dzdx_) ;
          test.eq(dzdw, dzdw_) ;
          test.eq(dzdb, dzdb_) ;
        end
      end
    end

  end
end
