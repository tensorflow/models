function v = setLayerOutputs(obj, layer, outputs)
%SETLAYEROUTPUTS  Set or change the outputs of a layer
%   Example: NET.SETLAYEROUTPUTS('layerName', {'output1', 'output2', ...})

v = [] ;
l = obj.getLayerIndex(layer) ;
for output = outputs
  v(end+1) = obj.addVar(char(output)) ;
end
obj.layers(l).outputs = outputs ;
obj.rebuild() ;
