function vl_bench_bnorm(gpu)
  if nargin < 1
    gpu = false ;
  end

  T = 100 ;
  x = randn(64,64,32,32,'single') ;
  g = randn(32,1,'single') ;
  b = randn(32,1,'single') ;

  if gpu
    x = gpuArray(x) ;
    g = gpuArray(g) ;
    b = gpuArray(b) ;
  end

  tic
  for t=1:T
    y = vl_nnbnorm(x,g,b) ;
  end
  if gpu, wait(gpuDevice) ; end
  fprintf('new: %f\n',toc);

  tic
  for t=1:T
    y_ = vl_nnbnorm_old(x,g,b) ;
  end
  if gpu, wait(gpuDevice) ; end
  fprintf('old: %f\n',toc);

  dzdy = randn(size(y),'single') ;
  if gpu
    dzdy = gpuArray(dzdy) ;
  end

  tic
  for t=1:T
    [a,b,c] = vl_nnbnorm(x,g,b,dzdy) ;
  end
  if gpu, wait(gpuDevice) ; end
  fprintf('new deriv: %f\n',toc);

  tic
  for t=1:T
    [a_,b_,c_] = vl_nnbnorm_old(x,g,b,dzdy) ;
  end
  if gpu, wait(gpuDevice) ; end
  fprintf('old deriv: %f\n',toc);

  vl_testsim(y,y_);
  vl_testsim(a,a_);
  vl_testsim(b,b_);
  vl_testsim(c,c_);
end
