function rebuild(obj)
%REBUILD Rebuild the internal data structures of a DagNN object
%   REBUILD(obj) rebuilds the internal data structures
%   of the DagNN obj. It is an helper function used internally
%   to update the network when layers are added or removed.

varFanIn = zeros(1, numel(obj.vars)) ;
varFanOut = zeros(1, numel(obj.vars)) ;
parFanOut = zeros(1, numel(obj.params)) ;

for l = 1:numel(obj.layers)
  ii = obj.getVarIndex(obj.layers(l).inputs) ;
  oi = obj.getVarIndex(obj.layers(l).outputs) ;
  pi = obj.getParamIndex(obj.layers(l).params) ;
  obj.layers(l).inputIndexes = ii ;
  obj.layers(l).outputIndexes = oi ;
  obj.layers(l).paramIndexes = pi ;
  varFanOut(ii) = varFanOut(ii) + 1 ;
  varFanIn(oi) = varFanIn(oi) + 1 ;
  parFanOut(pi) = parFanOut(pi) + 1 ;
end

[obj.vars.fanin] = tolist(num2cell(varFanIn)) ;
[obj.vars.fanout] = tolist(num2cell(varFanOut)) ;
if ~isempty(parFanOut)
  [obj.params.fanout] = tolist(num2cell(parFanOut)) ;
end

% dump unused variables
keep = (varFanIn + varFanOut) > 0 ;
obj.vars = obj.vars(keep) ;
varRemap = cumsum(keep) ;

% dump unused parameters
keep = parFanOut > 0 ;
obj.params = obj.params(keep) ;
parRemap = cumsum(keep) ;

% update the indexes to account for removed layers, variables and parameters
for l = 1:numel(obj.layers)
  obj.layers(l).inputIndexes = varRemap(obj.layers(l).inputIndexes) ;
  obj.layers(l).outputIndexes = varRemap(obj.layers(l).outputIndexes) ;
  obj.layers(l).paramIndexes = parRemap(obj.layers(l).paramIndexes) ;
  obj.layers(l).block.layerIndex = l ;
end

% update the variable and parameter names hash maps
obj.varNames = cell2struct(num2cell(1:numel(obj.vars)), {obj.vars.name}, 2) ;
obj.paramNames = cell2struct(num2cell(1:numel(obj.params)), {obj.params.name}, 2) ;
obj.layerNames = cell2struct(num2cell(1:numel(obj.layers)), {obj.layers.name}, 2) ;

% determine the execution order again (and check for consistency)
obj.executionOrder = getOrder(obj) ;

% --------------------------------------------------------------------
function order = getOrder(obj)
% --------------------------------------------------------------------
hops = cell(1, numel(obj.vars)) ;
for l = 1:numel(obj.layers)
  for v = obj.layers(l).inputIndexes
    hops{v}(end+1) = l ;
  end
end
order = zeros(1, numel(obj.layers)) ;
for l = 1:numel(obj.layers)
  if order(l) == 0
    order = dagSort(obj, hops, order, l) ;
  end
end
if any(order == -1)
  warning('The network graph contains a cycle') ;
end
[~,order] = sort(order, 'descend') ;

% --------------------------------------------------------------------
function order = dagSort(obj, hops, order, layer)
% --------------------------------------------------------------------
if order(layer) > 0, return ; end
order(layer) = -1 ; % mark as open
n = 0 ;
for o = obj.layers(layer).outputIndexes ;
  for child = hops{o}
    if order(child) == -1
      return ;
    end
    if order(child) == 0
      order = dagSort(obj, hops, order, child) ;
    end
    n = max(n, order(child)) ;
  end
end
order(layer) = n + 1 ;

% --------------------------------------------------------------------
function varargout = tolist(x)
% --------------------------------------------------------------------
[varargout{1:numel(x)}] = x{:} ;
