% VL_TEST_ECONOMIC_RELU
function vl_test_economic_relu()

x = randn(11,12,8,'single');
w = randn(5,6,8,9,'single');
b = randn(1,9,'single') ;

net.layers{1} = struct('type', 'conv', ...
                       'filters', w, ...
                       'biases', b, ...
                       'stride', 1, ...
                       'pad', 0);
net.layers{2} = struct('type', 'relu') ;

res = vl_simplenn(net, x) ;
dzdy = randn(size(res(end).x), 'like', res(end).x) ;
clear res ;

res_ = vl_simplenn(net, x, dzdy) ;
res__ = vl_simplenn(net, x, dzdy, [], 'conserveMemory', true) ;

a=whos('res_') ;
b=whos('res__') ;
assert(a.bytes > b.bytes) ;
vl_testsim(res_(1).dzdx,res__(1).dzdx,1e-4) ;
vl_testsim(res_(1).dzdw{1},res__(1).dzdw{1},1e-4) ;
vl_testsim(res_(1).dzdw{2},res__(1).dzdw{2},1e-4) ;
