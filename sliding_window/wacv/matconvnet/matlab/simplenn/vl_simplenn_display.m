function [info, str] = vl_simplenn_display(net, varargin)
%VL_SIMPLENN_DISPLAY  Display the structure of a SimpleNN network.
%   VL_SIMPLENN_DISPLAY(NET) prints statistics about the network NET.
%
%   INFO = VL_SIMPLENN_DISPLAY(NET) returns instead a structure INFO
%   with several statistics for each layer of the network NET.
%
%   [INFO, STR] = VL_SIMPLENN_DISPLAY(...) returns also a string STR
%   with the text that would otherwise be printed.
%
%   The function accepts the following options:
%
%   `inputSize`:: auto
%      Specifies the size of the input tensor X that will be passed to
%      the network as input. This information is used in order to
%      estiamte the memory required to process the network. When this
%      option is not used, VL_SIMPLENN_DISPLAY() tires to use values
%      in the NET structure to guess the input size:
%      NET.META.INPUTSIZE and NET.META.NORMALIZATION.IMAGESIZE
%      (assuming a batch size of one image, unless otherwise specified
%      by the `batchSize` option).
%
%   `batchSize`:: []
%      Specifies the number of data points in a batch in estimating
%      the memory consumption, overriding the last dimension of
%      `inputSize`.
%
%   `maxNumColumns`:: 18
%      Maximum number of columns in a table. Wider tables are broken
%      into multiple smaller ones.
%
%   `format`:: `'ascii'`
%      One of `'ascii'`, `'latex'`, or `'csv'`.
%
%   See also: VL_SIMPLENN().

% Copyright (C) 2014-15 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.inputSize = [] ;
opts.batchSize = [] ;
opts.maxNumColumns = 18 ;
opts.format = 'ascii' ;
opts = vl_argparse(opts, varargin) ;

% determine input size, using first the option, then net.meta.inputSize, 
% and eventually net.meta.normalization.imageSize, if any
if isempty(opts.inputSize)
  tmp = [] ;
  opts.inputSize = [NaN;NaN;NaN;1] ;
  if isfield(net, 'meta')
    if isfield(net.meta, 'inputSize')
      tmp =  net.meta.inputSize(:) ;
    elseif isfield(net.meta, 'normalization') && ...
        isfield(net.meta.normalization, 'imageSize')
      tmp = net.meta.normalization.imageSize ;
    end
    opts.inputSize(1:numel(tmp)) = tmp(:) ;
  end  
end

if ~isempty(opts.batchSize)
  opts.inputSize(4) = opts.batchSize ;
end


fields={'layer', 'type', 'name', '-', ...
        'support', 'filtd', 'filtdil', 'nfilt', 'stride', 'pad', '-', ...
        'rfsize', 'rfoffset', 'rfstride', '-', ...
        'dsize', 'ddepth', 'dnum', '-', ...
        'xmem', 'wmem'};

% get the support, stride, and padding of the operators
for l = 1:numel(net.layers)
  ly = net.layers{l} ;
  switch ly.type
    case 'conv'
      ks = max([size(ly.weights{1},1) ; size(ly.weights{1},2)],1) ;
      ks = (ks - 1) .* ly.dilate + 1 ;
      info.support(1:2,l) = ks ;
    case 'pool'
      info.support(1:2,l) = ly.pool(:) ;
    otherwise
      info.support(1:2,l) = [1;1] ;
  end
  if isfield(ly, 'stride')
    info.stride(1:2,l) = ly.stride(:) ;
  else
    info.stride(1:2,l) = 1 ;
  end
  if isfield(ly, 'pad')
    info.pad(1:4,l) = ly.pad(:) ;
  else
    info.pad(1:4,l) = 0 ;
  end

  % operator applied to the input image
  info.receptiveFieldSize(1:2,l) = 1 + ...
      sum(cumprod([[1;1], info.stride(1:2,1:l-1)],2) .* ...
          (info.support(1:2,1:l)-1),2) ;
  info.receptiveFieldOffset(1:2,l) = 1 + ...
      sum(cumprod([[1;1], info.stride(1:2,1:l-1)],2) .* ...
          ((info.support(1:2,1:l)-1)/2 - info.pad([1 3],1:l)),2) ;
  info.receptiveFieldStride = cumprod(info.stride,2) ;
end

% get the dimensions of the data
info.dataSize(1:4,1) = opts.inputSize(:) ;
for l = 1:numel(net.layers)
  ly = net.layers{l} ;
  if strcmp(ly.type, 'custom') && isfield(ly, 'getForwardSize')
    sz = ly.getForwardSize(ly, info.dataSize(:,l)) ;
    info.dataSize(:,l+1) = sz(:) ;
    continue ;
  end

  info.dataSize(1, l+1) = floor((info.dataSize(1,l) + ...
                                 sum(info.pad(1:2,l)) - ...
                                 info.support(1,l)) / info.stride(1,l)) + 1 ;
  info.dataSize(2, l+1) = floor((info.dataSize(2,l) + ...
                                 sum(info.pad(3:4,l)) - ...
                                 info.support(2,l)) / info.stride(2,l)) + 1 ;
  info.dataSize(3, l+1) = info.dataSize(3,l) ;
  info.dataSize(4, l+1) = info.dataSize(4,l) ;
  switch ly.type
    case 'conv'
      if isfield(ly, 'weights')
        f = ly.weights{1} ;
      else
        f = ly.filters ;
      end
      if size(f, 3) ~= 0
        info.dataSize(3, l+1) = size(f,4) ;
      end
    case {'loss', 'softmaxloss'}
      info.dataSize(3:4, l+1) = 1 ;
    case 'custom'
      info.dataSize(3,l+1) = NaN ;
  end
end

if nargout == 1, return ; end

% print table
table = {} ;
wmem = 0 ;
xmem = 0 ;
for wi=1:numel(fields)
  w = fields{wi} ;
  switch w
    case 'type', s = 'type' ;
    case 'stride', s = 'stride' ;
    case 'rfsize', s = 'rf size' ;
    case 'rfstride', s = 'rf stride' ;
    case 'rfoffset', s = 'rf offset' ;
    case 'dsize', s = 'data size' ;
    case 'ddepth', s = 'data depth' ;
    case 'dnum', s = 'data num' ;
    case 'nfilt', s = 'num filts' ;
    case 'filtd', s = 'filt dim' ;
    case 'filtdil', s = 'filt dilat' ;
    case 'wmem', s = 'param mem' ;
    case 'xmem', s = 'data mem' ;
    otherwise, s = char(w) ;
  end
  table{wi,1} = s ;

  % do input pseudo-layer
  for l=0:numel(net.layers)
    switch char(w)
      case '-', s='-' ;
      case 'layer', s=sprintf('%d', l) ;
      case 'dsize', s=pdims(info.dataSize(1:2,l+1)) ;
      case 'ddepth', s=sprintf('%d', info.dataSize(3,l+1)) ;
      case 'dnum', s=sprintf('%d', info.dataSize(4,l+1)) ;
      case 'xmem'
        a = prod(info.dataSize(:,l+1)) * 4 ;
        s = pmem(a) ;
        xmem = xmem + a ;
      otherwise
        if l == 0
          if strcmp(char(w),'type'), s = 'input';
          else s = 'n/a' ; end
        else
          ly=net.layers{l} ;
          switch char(w)
            case 'name'
              if isfield(ly, 'name')
                s=ly.name ;
              else
                s='' ;
              end
            case 'type'
              switch ly.type
                case 'normalize', s='norm';
                case 'pool'
                  if strcmpi(ly.method,'avg'), s='apool'; else s='mpool'; end
                case 'softmax', s='softmx' ;
                case 'softmaxloss', s='softmxl' ;
                otherwise s=ly.type ;
              end
            case 'nfilt'
              switch ly.type
                case 'conv'
                  if isfield(ly, 'weights'), a = size(ly.weights{1},4) ;
                  else, a = size(ly.filters,4) ; end
                  s=sprintf('%d',a) ;
                otherwise
                  s='n/a' ;
              end
            case 'filtd'
              switch ly.type
                case 'conv'
                  s=sprintf('%d',size(ly.weights{1},3)) ;
                otherwise
                  s='n/a' ;
              end
            case 'filtdil'
              switch ly.type
                case 'conv'
                  s=sprintf('%d',ly.dilate) ;
                otherwise
                  s='n/a' ;
              end

            case 'support'
              s = pdims(info.support(:,l)) ;
            case 'stride'
              s = pdims(info.stride(:,l)) ;
            case 'pad'
              s = pdims(info.pad(:,l)) ;
            case 'rfsize'
              s = pdims(info.receptiveFieldSize(:,l)) ;
            case 'rfoffset'
              s = pdims(info.receptiveFieldOffset(:,l)) ;
            case 'rfstride'
              s = pdims(info.receptiveFieldStride(:,l)) ;

            case 'wmem'
              a = 0 ;
              if isfield(ly, 'weights') ;
                for j=1:numel(ly.weights)
                  a = a + numel(ly.weights{j}) * 4 ;
                end
              end
              % Legacy code to be removed
              if isfield(ly, 'filters') ;
                a = a + numel(ly.filters) * 4 ;
              end
              if isfield(ly, 'biases') ;
                a = a + numel(ly.biases) * 4 ;
              end
              s = pmem(a) ;
              wmem = wmem + a ;
          end
        end
    end
    table{wi,l+2} = s ;
  end
end

str = {} ;
for i=2:opts.maxNumColumns:size(table,2)
  sel = i:min(i+opts.maxNumColumns-1,size(table,2)) ;
  str{end+1} = ptable(opts, table(:,[1 sel])) ;
end

table = {...
  'parameter memory', sprintf('%s (%.2g parameters)', pmem(wmem), wmem/4);
  'data memory', sprintf('%s (for batch size %d)', pmem(xmem), info.dataSize(4,1))} ;
str{end+1} = ptable(opts, table) ;

str = horzcat(str{:}) ;

if nargout == 0
  fprintf('%s', str) ;
  clear info str ;
end

% -------------------------------------------------------------------------
function str = ptable(opts, table)
% -------------------------------------------------------------------------
switch opts.format
  case 'ascii', str = pascii(table) ;
  case 'latex', str = platex(table) ;
  case 'csv',   str = pcsv(table) ; 
end  
str = horzcat(str,sprintf('\n')) ;

% -------------------------------------------------------------------------
function s = pmem(x)
% -------------------------------------------------------------------------
if isnan(x),       s = 'NaN' ;
elseif x < 1024^1, s = sprintf('%.0fB', x) ;
elseif x < 1024^2, s = sprintf('%.0fKB', x / 1024) ;
elseif x < 1024^3, s = sprintf('%.0fMB', x / 1024^2) ;
else               s = sprintf('%.0fGB', x / 1024^3) ;
end

% -------------------------------------------------------------------------
function s = pdims(x)
% -------------------------------------------------------------------------
if all(x==x(1))
  s = sprintf('%.4g', x(1)) ;
else
  s = sprintf('%.4gx', x(:)) ;
  s(end) = [] ;
end

% -------------------------------------------------------------------------
function str = pascii(table)
% -------------------------------------------------------------------------
str = {} ;
sizes = max(cellfun(@(x) numel(x), table),[],1) ;
for i=1:size(table,1)
  for j=1:size(table,2)
    s = table{i,j} ;
    fmt = sprintf('%%%ds|', sizes(j)) ;
    if isequal(s,'-'), s=repmat('-', 1, sizes(j)) ; end
    str{end+1} = sprintf(fmt, s) ;
  end
  str{end+1} = sprintf('\n') ;
end
str = horzcat(str{:}) ;

% -------------------------------------------------------------------------
function str = pcsv(table)
% -------------------------------------------------------------------------
str = {} ;
sizes = max(cellfun(@(x) numel(x), table),[],1) + 2 ;
for i=1:size(table,1)
  if isequal(table{i,1},'-'), continue ; end
  for j=1:size(table,2)
    s = table{i,j} ;
    str{end+1} = sprintf('%s,', ['"' s '"']) ;
  end
  str{end+1} = sprintf('\n') ;
end
str = horzcat(str{:}) ;

% -------------------------------------------------------------------------
function str = platex(table)
% -------------------------------------------------------------------------
str = {} ;
sizes = max(cellfun(@(x) numel(x), table),[],1) ;
str{end+1} = sprintf('\\begin{tabular}{%s}\n', repmat('c', 1, numel(sizes))) ;
for i=1:size(table,1)
  if isequal(table{i,1},'-'), str{end+1} = sprintf('\\hline\n') ; continue ; end
  for j=1:size(table,2)
    s = table{i,j} ;
    fmt = sprintf('%%%ds', sizes(j)) ;
    str{end+1} = sprintf(fmt, latexesc(s)) ;
    if j<size(table,2), str{end+1} = sprintf('&') ; end
  end
  str{end+1} = sprintf('\\\\\n') ;
end
str{end+1} = sprintf('\\end{tabular}\n') ;
str = horzcat(str{:}) ;

% -------------------------------------------------------------------------
function s = latexesc(s)
% -------------------------------------------------------------------------
s = strrep(s,'\','\\') ;
s = strrep(s,'_','\char`_') ;

% -------------------------------------------------------------------------
function [cpuMem,gpuMem] = xmem(s, cpuMem, gpuMem)
% -------------------------------------------------------------------------
if nargin <= 1
  cpuMem = 0 ;
  gpuMem = 0 ;
end
if isstruct(s)
  for f=fieldnames(s)'
    f = char(f) ;
    for i=1:numel(s)
      [cpuMem,gpuMem] = xmem(s(i).(f), cpuMem, gpuMem) ;
    end
  end
elseif iscell(s)
  for i=1:numel(s)
    [cpuMem,gpuMem] = xmem(s{i}, cpuMem, gpuMem) ;
  end
elseif isnumeric(s)
  if isa(s, 'single')
    mult = 4 ;
  else
    mult = 8 ;
  end
  if isa(s,'gpuArray')
    gpuMem = gpuMem + mult * numel(s) ;
  else
    cpuMem = cpuMem + mult * numel(s) ;
  end
end


