% Compare basic MatConvNet and Caffe blocks numerically
rootpath = fileparts(fileparts(mfilename('fullpath')));
run(fullfile(rootpath, 'matlab', 'vl_setupnn.m'));

inputScale = 100;

caffemodel = fullfile('data', 'tmp_caffemodels', 'test_model');
[~,~,~] = mkdir('data');
[~,~,~] = mkdir(fileparts(caffemodel));

%%
layers = {};
layers{end+1} = struct(...
  'name', 'conv', ...
  'type', 'conv', ...
  'stride', [2, 2], ...
  'pad', [1, 1, 1, 1], ...
  'weights', {{rand(3, 3, 10, 5, 'single'), rand(5, 1, 'single')}});

layers{end+1} = struct(...
  'name', 'relu', ...
  'type', 'relu');

layers{end+1} = struct(...
  'name', 'norm', ...
  'type', 'normalize', ...
  'param', [5, 1, 2e-5, 0.75]);

layers{end+1} = struct(...
  'name', 'softmax', ...
  'type', 'softmax');

%%
net_ = struct();
net_.meta.normalization.imageSize = [20, 20, 10];
net_.meta.normalization.averageImage = rand(1, 1, 10);

diffStats = zeros(numel(layers), 3);
for li = 1:numel(layers)
  net_.layers = layers(li);
  layerName = layers{li}.name;
  simplenn_caffe_deploy(net_, caffemodel, 'doTest', false, ...
    'outputBlobName', layerName, 'silent', true);
  res = simplenn_caffe_compare(net_, caffemodel, [], ...
    'randScale', inputScale, 'silent', true);
  diffStats(li, :) = res.(layerName);
end

fprintf('Results: \n');
layerNames = cellfun(@(l) l.name, layers, 'UniformOutput', false);
fprintf('Layer   %s\n', sprintf('% 10s', layerNames{:}));
fprintf('MeanErr %s\n', sprintf('% 10.2e', diffStats(:, 2)));
fprintf('MaxErr  %s\n', sprintf('% 10.2e', diffStats(:, 3)));

rmdir(fileparts(caffemodel), 's');
