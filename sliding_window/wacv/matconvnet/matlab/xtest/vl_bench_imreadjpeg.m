% VL_BENCH_IMREADJPEG   Evaluates the speed of imreadjpeg

numThreads = 4 ;
base = 'data/bench-imreadjpeg' ;

files = {} ;
files = dir(fullfile(base,'*.jpg')) ;
files = fullfile(base, {files.name}) ;
if numel(files) > 256, files = files(1:256) ; end

for preallocate = [true, false]
  opts={'verbose','verbose', 'preallocate', preallocate} ;
  for t=1:4
    % simple read
    fprintf('direct read single thread\n') ;
    clear ims ;
    tic  ;
    ims = vl_imreadjpeg(files, 'numThreads', 1, opts{:}) ;
    directSingle(t) = toc ;
    fprintf('   done\n') ;
    pause(1) ;

    % simple read
    fprintf('direct read multi thread\n') ;
    clear ims ;
    tic  ;
    ims = vl_imreadjpeg(files, 'numThreads', numThreads, opts{:}) ;
    direct(t) = toc ;
    fprintf('   done\n') ;
    pause(1) ;

    % threaded read
    fprintf('issue prefetch\n') ;
    tic ;
    vl_imreadjpeg(files, 'prefetch', opts{:}) ;
    prefetch(t) = toc ;
    fprintf('   done [pause 6]\n') ;
    pause(6)

    fprintf('prefetched read\n') ;
    clear ims_ ; % do not accoutn for the time requried to delete this
    tic ;
    ims_ = vl_imreadjpeg(files, opts{:}) ;
    indirect(t) = toc ;
    pause(1) ;
  end

  n = numel(ims) ;
  fprintf('** test results preallcoate %d\n', preallocate) ;
  fprintf('\tsingle tread: %.1f pm %.1f\n', mean(n./directSingle), std(n./directSingle)) ;
  fprintf('\t%d threads: %.1f pm %.1f\n', numThreads, mean(n./direct), std(n./direct)) ;
  fprintf('\tissue prefetch: %.1f pm %.1f\n', mean(n./prefetch), std(n./prefetch)) ;
  fprintf('\tretrieve prefetched: %.1f pm %.1f\n', mean(n./indirect), std(n./indirect)) ;
  fprintf('\n\n') ;
end

return
