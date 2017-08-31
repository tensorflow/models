% function [net, info] = xnet_cnn(param,epochs)

function [net, info] = xnet_cnn(var1,var2,cnnmode)

if strcmp(cnnmode,'train')==1
    param  = var1;
    epochs = var2;
    train  = true;
else
    info   = var2;
    train  = false;
    epochs = info.opts.train.numEpochs;
    param  = info.param;
end

basepath = '';
opts.dataDir = fullfile(basepath) ;
opts.modelType = 'xnet' ;
opts.useGpu = false ;
opts.networkType = 'dagnn';
% sfx = opts.modelType ;
opts.expDir = fullfile(opts.dataDir, 'epochs');
opts.numFetchThreads = 60;
opts.lite = false;
opts.batchSize = 256;

% opts.imdbPath = fullfile(opts.dataDir,'imdb.mat');
opts.imdbPath = fullfile(opts.dataDir,'../imdb.mat');

opts.train.prefetch = false;
opts.model.nChannels = 1;
opts.model.colorSpace  = 'gray';
opts.train.gpus = [];
opts.train.numEpochs = epochs;
opts.train.learningRate = logspace(-2, -4, 60);
%opts.train.derOutputs = {'top1err', 1,'top5err', 1} ;

% -------------------------------------------------------------------------
%                                                             Prepare model
% -------------------------------------------------------------------------

net = xnet_init(param);
% pretrain = fullfile(basepath,'minc-2500','vgg_s_fromINetGray');
% modelPath = @(ep) fullfile(pretrain, sprintf('net-epoch-%d.mat', ep));
% epoch = 20;
% fprintf('loading pretrained model at epoch %d\n', epoch);
% load(modelPath(epoch), 'net');
% net = dagnn.DagNN.loadobj(net);

%net.meta.augmentation.transformation = 'f5';

% -------------------------------------------------------------------------
%                                                              Prepare data
% -------------------------------------------------------------------------

load(opts.imdbPath);

if exist(opts.expDir,'dir')==0
    mkdir(opts.expDir) ;
end

% Set the class names in the network
net.meta.classes.name = imdb.meta.classes;
imdb.images.class = [];
imdb.meta.normalization.averageImage = [];

% Compute image statistics (mean, RGB covariances, etc.)
imageStatsPath = fullfile(opts.expDir, 'imageStats.mat') ;
n = 'xnet_data';
if exist(imageStatsPath,'var')
    load(n, 'averageImage', 'rgbMean', 'rgbCovariance') ;
else
    [averageImage, rgbMean, rgbCovariance] = getImageStats(opts, net.meta, imdb) ;
    save(n, 'averageImage', 'rgbMean', 'rgbCovariance') ;
end

% Set the image average (use either an image or a color)
%net.meta.normalization.averageImage = averageImage ;
net.meta.normalization.averageImage = rgbMean ;
imdb.meta.normalization.averageImage = rgbMean ;

if train
    % Set data augmentation statistics
    [v,d] = eig(rgbCovariance) ;
    net.meta.augmentation.rgbVariance = 0.1*sqrt(d)*v' ;
    clear v d ;
    
    % -------------------------------------------------------------------------
    %                                                                     Train
    % -------------------------------------------------------------------------
    [net, info] = cnn_train_dag(net, imdb, getBatchFn(opts, net.meta), ...
        'expDir', opts.expDir, ...
        net.meta.trainOpts,...
        opts.train);
    
    % -------------------------------------------------------------------------
    %                                                                    Deploy
    % -------------------------------------------------------------------------
    net.removeLayer('loss');
    net.removeLayer('top1err');
    net.addLayer('softmax', ...
        dagnn.SoftMax(), ...
        {'prediction','label'}, 'preddist');
    
    info.opts = opts;
    info.param = param;
    
else
    
    % -------------------------------------------------------------------------
    %                                                                      Test
    % -------------------------------------------------------------------------
    % net.move('gpu');
    net = var1;
    opts = info.opts;
    
    net.mode = 'test';
    useGpu = numel(opts.train.gpus) > 0;
    testset = find(imdb.images.set==3);
    labels = imdb.images.label(testset);
    classError = zeros(numel(testset),1);
    Prediction = zeros(numel(testset),1);
    for b = 1:opts.batchSize:numel(testset)
        batch = testset(b:min(b+opts.batchSize-1,numel(testset)));
        inputs = getDagNNBatch(opts, useGpu, imdb, batch);
        net.eval(inputs) ;
        pred = gather(net.vars(net.getVarIndex('preddist')).value) ;
        [~,predClass] = max(pred,[],3);
        predClass = reshape(predClass,size(predClass,4),1);
        classError(b:min(b+opts.batchSize-1,numel(testset))) = labels(b:min(b+opts.batchSize-1,numel(testset)))' ~= predClass;
        Prediction(b:min(b+opts.batchSize-1,numel(testset))) = predClass;
    end
    
    info.acc = 100-nnz(classError)*100/numel(testset);
    info.ds  = Prediction;
    info.C   = Bev_confusion(Prediction,labels');
    info.p   = Bev_performance(Prediction,labels');
    
    
%     disp(' ');
%     disp('--------------------');
%     disp(['Accuracy = ' num2str(info.acc) '%']);
%     disp('--------------------');
%     disp(' ');
end

% -------------------------------------------------------------------------
function fn = getBatchFn(opts, meta)
% -------------------------------------------------------------------------
useGpu = numel(opts.train.gpus) > 0 ;

bopts.numThreads = opts.numFetchThreads ;
bopts.imageSize = meta.normalization.imageSize ;
bopts.border = meta.normalization.border ;
bopts.averageImage = meta.normalization.averageImage ;
bopts.rgbVariance = meta.augmentation.rgbVariance ;
bopts.transformation = meta.augmentation.transformation ;
bopts.colorSpace = opts.model.colorSpace;

fn = @(x,y) getDagNNBatch(bopts,useGpu,x,y) ;

% -------------------------------------------------------------------------
function inputs = getDagNNBatch(opts, useGpu, imdb, batch)
% -------------------------------------------------------------------------
images = imdb.images.data(:,:,batch);% strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
isVal = ~isempty(batch) && imdb.images.set(batch(1)) ~= 1 ;

im = zeros(size(images,1), size(images,2), 1, size(images,3),'single');

if ~isempty(imdb.meta.normalization.averageImage)
    for i = 1:size(images,3)
        im(:,:,:,i) = 255*images(:,:,i) - imdb.meta.normalization.averageImage;
    end
else
    for i = 1:size(images,3)
        im(:,:,:,i) = 255*images(:,:,i);
    end
end

% if ~isVal
%   % training
%   im = cnn_xray_get_batch(images, opts, ...
%                               'prefetch', nargout == 0) ;
% else
%   % validation: disable data augmentation
%   im = cnn_xray_get_batch(images, opts, ...
%                               'prefetch', nargout == 0, ...
%                               'transformation', 'none') ;
% end

if nargout > 0
    if useGpu
        im = gpuArray(im) ;
    end
    labels = imdb.images.label(batch) ;
    inputs = {'input', im, 'label', labels} ;
end

% -------------------------------------------------------------------------
function [averageImage, rgbMean, rgbCovariance] = getImageStats(opts, meta, imdb)
% -------------------------------------------------------------------------
train = find(imdb.images.set == 1) ;
train = train(1:end);
bs = 256 ;
opts.train.colorSpace  = 'rgb';
fn = getBatchFn(opts, meta) ;
avg = {}; rgbm1 = {}; rgbm2 = {};

for t=1:bs:numel(train)
    batch_time = tic ;
    batch = train(t:min(t+bs-1, numel(train))) ;
    % fprintf('collecting image stats: batch starting with image %d ...', batch(1)) ;
    temp = fn(imdb, batch) ;
    temp = gather(temp{2});
    z = reshape(permute(temp,[3 1 2 4]),1,[]) ;
    n = size(z,2) ;
    avg{end+1} = mean(temp, 4) ;
    rgbm1{end+1} = sum(z,2)/n ;
    rgbm2{end+1} = z*z'/n ;
    batch_time = toc(batch_time) ;
    % fprintf(' %.2f s (%.1f images/s)\n', batch_time, numel(batch)/ batch_time) ;
end
averageImage = mean(cat(4,avg{:}),4) ;
rgbm1 = mean(cat(2,rgbm1{:}),2) ;
rgbm2 = mean(cat(3,rgbm2{:}),3) ;
rgbMean = rgbm1 ;
rgbCovariance = rgbm2 - rgbm1*rgbm1' ;
