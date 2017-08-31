function vl_test_print(varargin)

addpath(fullfile(vl_rootnn(), 'examples', 'mnist'));

net = cnn_mnist_init('networkType', 'dagnn');
net.print(varargin{:});

end

