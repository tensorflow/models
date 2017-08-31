function vl_simplenn_start_parserv(net, ps)
%VL_SIMPLENN_START_PARSERV   Setup a parameter server for this network
%    VL_SIMPLENN_START_PARSERV(NET, PS) registers the network
%    parameter derivatives with the specified ParameterServer instance
%    PS and then starts the server.

for i = 1:numel(net.layers)
  for j = 1:numel(net.layers{i}.weights)
    value = net.layers{i}.weights{j} ;
    name = sprintf('l%d_%d',i,j) ;
    if strcmp(class(value),'gpuArray')
      deviceType = 'gpu' ;
      dataType = classUnderlying(value) ;
    else
      deviceType = 'cpu' ;
      dataType = class(value) ;
    end
    ps.register(...
      name, ...
      size(value), ...
      dataType, ...
      deviceType) ;
  end
end
ps.start() ;
