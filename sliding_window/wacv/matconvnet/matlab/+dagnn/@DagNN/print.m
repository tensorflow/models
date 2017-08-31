function str = print(obj, inputSizes, varargin)
%PRINT Print information about the DagNN object
%   PRINT(OBJ) displays a summary of the functions and parameters in the network.
%   STR = PRINT(OBJ) returns the summary as a string instead of printing it.
%
%   PRINT(OBJ, INPUTSIZES) where INPUTSIZES is a cell array of the type
%   {'input1nam', input1size, 'input2name', input2size, ...} prints
%   information using the specified size for each of the listed inputs.
%
%   PRINT(___, 'OPT', VAL, ...) accepts the following options:
%
%   `All`:: false
%      Display all the information below.
%
%   `Layers`:: '*'
%      Specify which layers to print. This can be either a list of
%      indexes, a cell array of array names, or the string '*', meaning
%      all layers.
%
%   `Parameters`:: '*'
%      Specify which parameters to print, similar to the option above.
%
%   `Variables`:: []
%      Specify which variables to print, similar to the option above.
%
%   `Dependencies`:: false
%      Whether to display the dependency (geometric transformation)
%      of each variables from each input.
%
%   `Format`:: 'ascii'
%      Choose between `ascii`, `latex`, `csv`, 'digraph', and `dot`.
%      The first three format print tables; `digraph` uses the plot function
%      for a `digraph` (supported in MATLAB>=R2015b) and the last one
%      prints a graph  in `dot` format. In case of zero outputs, it
%      attmepts to compile and visualise the dot graph using `dot` command
%      and `start` (Windows), `display` (Linux) or `open` (Mac OSX) on your system.
%      In the latter case, all variables and layers are included in the
%      graph, regardless of the other parameters.
%
%   `FigurePath`:: 'tempname.pdf'
%      Sets the path where any generated `dot` figure will be saved. Currently,
%      this is useful only in combination with the format `dot`. 
%      By default, a unique temporary filename is used (`tempname`
%      is replaced with a `tempname()` call). The extension specifies the
%      output format (passed to dot as a `-Text` parameter).
%      If not extension provided, PDF used by default.
%      Additionally, stores the .dot file used to generate the figure to
%      the same location.
%
%    `dotArgs`:: ''
%       Additional dot arguments. E.g. '-Gsize="7"' to generate a smaller
%       output (for a review of the network structure etc.).
%
%   `MaxNumColumns`:: 18
%      Maximum number of columns in each table.
%
%   See also: DAGNN, DAGNN.GETVARSIZES().

if nargin > 1 && ischar(inputSizes)
  % called directly with options, skipping second argument
  varargin = {inputSizes, varargin{:}} ;
  inputSizes = {} ;
end

opts.all = false ;
opts.format = 'ascii' ;
opts.figurePath = 'tempname.pdf' ;
opts.dotArgs = '';
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.layers = '*' ;
opts.parameters = [] ;
opts.variables = [] ;
if opts.all || nargin > 1
  opts.variables = '*' ;
end
if opts.all
  opts.parameters = '*' ;
end
opts.memory = true ;
opts.dependencies = opts.all ;
opts.maxNumColumns = 18 ;
opts = vl_argparse(opts, varargin) ;

if nargin == 1, inputSizes = {} ; end
varSizes = obj.getVarSizes(inputSizes) ;
paramSizes = cellfun(@size, {obj.params.value}, 'UniformOutput', false) ;
str = {''} ;

if strcmpi(opts.format, 'dot')
  str = printDot(obj, varSizes, paramSizes, opts) ;
  if nargout == 0
    displayDot(str, opts) ;
  end
  return ;
end

if strcmpi(opts.format,'digraph')
  str = printdigraph(obj, varSizes) ;
  return ;
end

if ~isempty(opts.layers)
  table = {'func', '-', 'type', 'inputs', 'outputs', 'params', 'pad', 'stride'} ;
  for l = select(obj, 'layers', opts.layers)
    layer = obj.layers(l) ;
    table{l+1,1} = layer.name ;
    table{l+1,2} = '-' ;
    table{l+1,3} = player(class(layer.block)) ;
    table{l+1,4} = strtrim(sprintf('%s ', layer.inputs{:})) ;
    table{l+1,5} = strtrim(sprintf('%s ', layer.outputs{:})) ;
    table{l+1,6} = strtrim(sprintf('%s ', layer.params{:})) ;
    if isprop(layer.block, 'pad')
      table{l+1,7} = pdims(layer.block.pad) ;
    else
      table{l+1,7} = 'n/a' ;
    end
    if isprop(layer.block, 'stride')
      table{l+1,8} = pdims(layer.block.stride) ;
    else
      table{l+1,8} = 'n/a' ;
    end
  end
  str{end+1} = printtable(opts, table') ;
  str{end+1} = sprintf('\n') ;
end

if ~isempty(opts.parameters)
  table = {'param', '-', 'dims', 'mem', 'fanout'} ;
  for v = select(obj, 'params', opts.parameters)
    table{v+1,1} = obj.params(v).name ;
    table{v+1,2} = '-' ;
    table{v+1,3} = pdims(paramSizes{v}) ;
    table{v+1,4} = pmem(prod(paramSizes{v}) * 4) ;
    table{v+1,5} = sprintf('%d',obj.params(v).fanout) ;
  end
  str{end+1} = printtable(opts, table') ;
  str{end+1} = sprintf('\n') ;
end

if ~isempty(opts.variables)
  table = {'var', '-', 'dims', 'mem', 'fanin', 'fanout'} ;
  for v = select(obj, 'vars', opts.variables)
    table{v+1,1} = obj.vars(v).name ;
    table{v+1,2} = '-' ;
    table{v+1,3} = pdims(varSizes{v}) ;
    table{v+1,4} = pmem(prod(varSizes{v}) * 4) ;
    table{v+1,5} = sprintf('%d',obj.vars(v).fanin) ;
    table{v+1,6} = sprintf('%d',obj.vars(v).fanout) ;
  end
  str{end+1} = printtable(opts, table') ;
  str{end+1} = sprintf('\n') ;
end

if opts.memory
  paramMem = sum(cellfun(@getMem, paramSizes)) ;
  varMem = sum(cellfun(@getMem, varSizes)) ;
  table = {'params', 'vars', 'total'} ;
  table{2,1} = pmem(paramMem) ;
  table{2,2} = pmem(varMem) ;
  table{2,3} = pmem(paramMem + varMem) ;
  str{end+1} = printtable(opts, table') ;
  str{end+1} = sprintf('\n') ;
end

if opts.dependencies
  % print variable to input dependencies
  inputs = obj.getInputs() ;
  rfs = obj.getVarReceptiveFields(inputs) ;
  for i = 1:size(rfs,1)
    table = {sprintf('rf in ''%s''', inputs{i}), '-', 'size', 'stride', 'offset'} ;
    for v = 1:size(rfs,2)
      table{v+1,1} = obj.vars(v).name ;
      table{v+1,2} = '-' ;
      table{v+1,3} = pdims(rfs(i,v).size) ;
      table{v+1,4} = pdims(rfs(i,v).stride) ;
      table{v+1,5} = pdims(rfs(i,v).offset) ;
    end
    str{end+1} = printtable(opts, table') ;
    str{end+1} = sprintf('\n') ;
  end
end

% finish
str = horzcat(str{:}) ;
if nargout == 0,
  fprintf('%s',str) ;
  clear str ;
end

end

% -------------------------------------------------------------------------
function str = printtable(opts, table)
% -------------------------------------------------------------------------
str = {''} ;
for i=2:opts.maxNumColumns:size(table,2)
  sel = i:min(i+opts.maxNumColumns-1,size(table,2)) ;
  str{end+1} = printtablechunk(opts, table(:, [1 sel])) ;
  str{end+1} = sprintf('\n') ;
end
str = horzcat(str{:}) ;
end

% -------------------------------------------------------------------------
function str = printtablechunk(opts, table)
% -------------------------------------------------------------------------
str = {''} ;
switch opts.format
  case 'ascii'
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

  case 'latex'
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
    str{end+1}= sprintf('\\end{tabular}\n') ;

  case 'csv'
    sizes = max(cellfun(@(x) numel(x), table),[],1) + 2 ;
    for i=1:size(table,1)
      if isequal(table{i,1},'-'), continue ; end
      for j=1:size(table,2)
        s = table{i,j} ;
        fmt = sprintf('%%%ds,', sizes(j)) ;
        str{end+1} = sprintf(fmt, ['"' s '"']) ;
      end
      str{end+1} = sprintf('\n') ;
    end

  otherwise
    error('Uknown format %s', opts.format) ;
end
str = horzcat(str{:}) ;
end

% -------------------------------------------------------------------------
function s = latexesc(s)
% -------------------------------------------------------------------------
s = strrep(s,'\','\\') ;
s = strrep(s,'_','\char`_') ;
end

% -------------------------------------------------------------------------
function s = pmem(x)
% -------------------------------------------------------------------------
if isnan(x),       s = 'NaN' ;
elseif x < 1024^1, s = sprintf('%.0fB', x) ;
elseif x < 1024^2, s = sprintf('%.0fKB', x / 1024) ;
elseif x < 1024^3, s = sprintf('%.0fMB', x / 1024^2) ;
else               s = sprintf('%.0fGB', x / 1024^3) ;
end
end

% -------------------------------------------------------------------------
function s = pdims(x)
% -------------------------------------------------------------------------
if all(isnan(x))
  s = 'n/a' ;
  return ;
end
if all(x==x(1))
  s = sprintf('%.4g', x(1)) ;
else
  s = sprintf('%.4gx', x(:)) ;
  s(end) = [] ;
end
end

% -------------------------------------------------------------------------
function x = player(x)
% -------------------------------------------------------------------------
if numel(x) < 7, return ; end
if x(1:6) == 'dagnn.', x = x(7:end) ; end
end

% -------------------------------------------------------------------------
function m = getMem(sz)
% -------------------------------------------------------------------------
m = prod(sz) * 4 ;
if isnan(m), m = 0 ; end
end

% -------------------------------------------------------------------------
function sel = select(obj, type, pattern)
% -------------------------------------------------------------------------
if isnumeric(pattern)
  sel = pattern ;
else
  if isstr(pattern)
    if strcmp(pattern, '*')
      sel = 1:numel(obj.(type)) ;
      return ;
    else
      pattern = {pattern} ;
    end
  end
  sel = find(cellfun(@(x) any(strcmp(x, pattern)), {obj.(type).name})) ;
end
end

% -------------------------------------------------------------------------
function h = printdigraph(net, varSizes)
% -------------------------------------------------------------------------
if exist('digraph') ~= 2
  error('MATLAB graph support not present.');
end
s = []; t = []; w = [];
varsNames = {net.vars.name};
layerNames = {net.layers.name};
numVars = numel(varsNames);
spatSize = cellfun(@(vs) vs(1), varSizes);
spatSize(isnan(spatSize)) = 1;
varChannels = cellfun(@(vs) vs(3), varSizes);
varChannels(isnan(varChannels)) = 0;

for li = 1:numel(layerNames)
  l = net.layers(li); lidx = numVars + li;
  s = [s l.inputIndexes];
  t = [t lidx*ones(1, numel(l.inputIndexes))];
  w = [w spatSize(l.inputIndexes)];
  s = [s lidx*ones(1, numel(l.outputIndexes))];
  t = [t l.outputIndexes];
  w = [w spatSize(l.outputIndexes)];
end
nodeNames = [varsNames, layerNames];
g = digraph(s, t, w);
lw = 5*g.Edges.Weight/max([g.Edges.Weight; 5]);
h = plot(g, 'NodeLabel', nodeNames, 'LineWidth', lw);
highlight(h, numVars+1:numVars+numel(layerNames), 'MarkerSize', 8, 'Marker', 's');
highlight(h, 1:numVars, 'MarkerSize', 5, 'Marker', 's');
cmap = copper;
varNvalRel = varChannels./max(varChannels);
for vi = 1:numel(varChannels)
  highlight(h, vi, 'NodeColor', cmap(max(round(varNvalRel(vi)*64), 1),:));
end
axis off;
layout(h, 'force');
end

% -------------------------------------------------------------------------
function str = printDot(net, varSizes, paramSizes, otps)
% -------------------------------------------------------------------------
str = {} ;
str{end+1} = sprintf('digraph DagNN {\n\tfontsize=12\n') ;
font_style = 'fontsize=12 fontname="helvetica"';

for v = 1:numel(net.vars)
  label=sprintf('{{%s} | {%s | %s }}', net.vars(v).name, pdims(varSizes{v}), pmem(4*prod(varSizes{v}))) ;
  str{end+1} = sprintf('\tvar_%s [label="%s" shape=record style="solid,rounded,filled" color=cornsilk4 fillcolor=beige %s ]\n', ...
    net.vars(v).name, label, font_style) ;
end

for p = 1:numel(net.params)
  label=sprintf('{{%s} | {%s | %s }}', net.params(p).name, pdims(paramSizes{p}), pmem(4*prod(paramSizes{p}))) ;
  str{end+1} = sprintf('\tpar_%s [label="%s" shape=record style="solid,rounded,filled" color=lightsteelblue4 fillcolor=lightsteelblue %s ]\n', ...
    net.params(p).name, label, font_style) ;
end

for l = 1:numel(net.layers)
  label = sprintf('{ %s | %s }', net.layers(l).name, class(net.layers(l).block)) ;
  str{end+1} = sprintf('\t%s [label="%s" shape=record style="bold,filled" color="tomato4" fillcolor="tomato" %s ]\n', ...
    net.layers(l).name, label, font_style) ;
  for i = 1:numel(net.layers(l).inputs)
    str{end+1} = sprintf('\tvar_%s->%s [weight=10]\n', ...
      net.layers(l).inputs{i}, ...
      net.layers(l).name) ;
  end
  for o = 1:numel(net.layers(l).outputs)
    str{end+1} = sprintf('\t%s->var_%s [weight=10]\n', ...
      net.layers(l).name, ...
      net.layers(l).outputs{o}) ;
  end
  for p = 1:numel(net.layers(l).params)
    str{end+1} = sprintf('\tpar_%s->%s [weight=1]\n', ...
      net.layers(l).params{p}, ...
      net.layers(l).name) ;
  end
end

str{end+1} = sprintf('}\n') ;
str = cat(2,str{:}) ;
end

% -------------------------------------------------------------------------
function displayDot(str, opts)
% -------------------------------------------------------------------------
%mwdot = fullfile(matlabroot, 'bin', computer('arch'), 'mwdot') ;
dotPaths = {'/opt/local/bin/dot', 'dot'} ;
if ismember(computer, {'PCWIN64', 'PCWIN'})
  winPath = 'c:\Program Files (x86)';
  dpath = dir(fullfile(winPath, 'Graphviz*'));
  if ~isempty(dpath)
    dotPaths = [{fullfile(winPath, dpath.name, 'bin', 'dot.exe')}, dotPaths];
  end
end
dotExe = '' ;
for i = 1:numel(dotPaths)
  [~,~,ext] = fileparts(dotPaths{i});
  if exist(dotPaths{i},'file') && ~strcmp(ext, '.m')
    dotExe = dotPaths{i} ;
    break;
  end
end
if isempty(dotExe)
  warning('Could not genereate a figure because the `dot` utility could not be found.') ;
  return ;
end

[path, figName, ext] = fileparts(opts.figurePath) ;

if isempty(ext), ext = '.pdf' ; end
if strcmp(figName, 'tempname')
  figName = tempname();
end
in = fullfile(path, [ figName, '.dot' ]) ;
out = fullfile(path, [ figName, ext ]) ;

f = fopen(in, 'w') ; fwrite(f, str) ; fclose(f) ;

cmd = sprintf('"%s" -T%s %s -o "%s" "%s"', dotExe, ext(2:end), ...
  opts.dotArgs, out, in) ;
[status, result] = system(cmd) ;
if status ~= 0
  error('Unable to run %s\n%s', cmd, result) ;
end
if ~isempty(strtrim(result))
  fprintf('Dot output:\n%s\n', result) ;
end

%f = fopen(out,'r') ; file=fread(f, 'char=>char')' ; fclose(f) ;
switch computer
  case {'PCWIN64', 'PCWIN'}
    system(sprintf('start "" "%s"', out)) ;
  case 'MACI64'
    system(sprintf('open "%s"', out)) ;
  case 'GLNXA64'
    system(sprintf('display "%s"', out)) ;
  otherwise
    fprintf('The figure saved at "%s"\n', out) ;
end
end
