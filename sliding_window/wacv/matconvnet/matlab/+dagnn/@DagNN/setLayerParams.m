function v = setLayerParams(obj, layer, params)
%SETLAYEPARAMS  Set or change the parameters of a layer
%   Example: NET.SETLAYERPARAMS('layerName', {'param1', 'param2', ...})

v = [] ;
l = obj.getLayerIndex(layer) ;
for param = params
  v(end+1) = obj.addParam(char(param)) ;
end
obj.layers(l).params = params ;
obj.rebuild() ;
