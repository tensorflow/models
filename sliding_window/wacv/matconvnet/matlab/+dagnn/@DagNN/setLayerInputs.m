function v = setLayerInputs(obj, layer, inputs)
%SETLAYERINPUTS  Set or change the inputs to a layer
%   Example: NET.SETLAYERINPUTS('layerName', {'input1', 'input2', ...})

v = [] ;
l = obj.getLayerIndex(layer) ;
for input = inputs
  v(end+1) = obj.addVar(char(input)) ;
end
obj.layers(l).inputs = inputs ;
obj.rebuild() ;
