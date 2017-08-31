function vl_test_imreadjpeg
% VL_TEST_IMREADJPEG

% Test basic file reading capability
for t=1:6
  files{t} = which(sprintf('office_%d.jpg', t)) ;
end
ims = vl_imreadjpeg(files) ;

% Test reading a CMYK image
ims_cmyk = vl_imreadjpeg({which('cmyk.jpg')}) ;

ims = vl_imreadjpeg(files) ;
assert(all(~cellfun(@isempty, ims)), 'Imagae Files not loaded.');

% Test inserting a non-image file
files_ = files ;
files_{3} = [mfilename('fullpath') '.m'];
ims_ = vl_imreadjpeg(files_) ;
for t=setdiff(1:6,3)
  assert(isequal(ims{t},ims_{t})) ;
end

% Test inserting a non-esiting file
files__ = files_ ;
files__{4} = 'idontexist.jpg' ;
ims__ = vl_imreadjpeg(files__) ;
for t=setdiff(1:6,[3 4])
  assert(isequal(ims{t},ims__{t})) ;
end

for n = 1:4
  % Test prefetching
  vl_imreadjpeg(files,'prefetch', 'numThreads', n) ;
  ims___ = vl_imreadjpeg(files) ;
  assert(isequal(ims,ims___)) ;

  % Hardening: test prefetching, clearing mex, fetching
  vl_imreadjpeg(files,'prefetch') ;
  clear mex ;
  ims___ = vl_imreadjpeg(files, 'numThreads', n) ;
  assert(isequal(ims,ims___)) ;
end

ims = vl_imreadjpeg(files) ;
