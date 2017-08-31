function opfx = wacv_fxdef(feature,im)


opfx.fx = feature;

% for deep features only
[N,M]       = size(im);
im3         = zeros(N,M,3);
im3(:,:,1)  = im;
im3(:,:,2)  = im;
im3(:,:,3)  = im;
im_         = single(im3) ;


switch lower(feature)
    case 'lbp'
        opfx.vdiv = 1;                  % one vertical divition
        opfx.hdiv = 1;                  % one horizontal divition
        opfx.semantic = 0;              % classic LBP
        opfx.samples  = 8;              % number of neighbor samples
        opfx.mappingtype = 'u2';        % uniform LBP
        opfx.m = 59;
        opfx.method = 2;
    case 'lbpri'
        opfx.vdiv = 1;                  % one vertical divition
        opfx.hdiv = 1;                  % one horizontal divition
        opfx.semantic = 0;              % classic LBP
        opfx.samples  = 8;              % number of neighbor samples
        opfx.mappingtype = 'ri';        % rotation invariant
        opfx.m = 36;
        opfx.method = 2;
    case 'slbp'
        opfx.vdiv = 1;                  % one vertical divition
        opfx.hdiv = 1;                  % one horizontal divition
        opfx.semantic = 1;              % semantic LBP
        opfx.samples  = 16;             % number of neighbor samples
        opfx.sk       = 0.5;
        opfx.m = 123;
        opfx.method = 2;
    case 'bsif'
        nbits      = 8;
        opfx.vdiv = 1;                  % one vertical divition
        opfx.hdiv = 1;                  % one horizontal divition
        opfx.filter  = 3;               % use filter 5x5
        opfx.bits  = nbits;             % use nbits bits filter
        opfx.mode  = 'h';
        opfx.m = 2^nbits;
        opfx.method = 3;
    case {'txh','haralick'}
        opfx.dharalick = 1:2:5;
        opfx.m = 28*length(opfx.dharalick);
        opfx.fx = 'txh';
        opfx.method = 4;
    case {'dft','fft'}
        opfx.m = round(N/2)*round(M/2);
        opfx.fx = 'dft';
        opfx.method = 12;
    case 'dct'
        opfx.m = N*M;
        opfx.fx = 'dct';
        opfx.method = 17;
    case 'clp'
        opfx.m = 12;
        opfx.ng = 32;
        opfx.show = 0;
        opfx.method = 5;
    case {'gab','gabor'}
        r          = 4;
        opfx.m       = r^2+3;
        opfx.Lgabor  = r;                 % number of rotations
        opfx.Sgabor  = r;                 % number of dilations (scale)
        opfx.fhgabor = 2;                 % highest frequency of interest
        opfx.flgabor = 0.1;               % lowest frequency of interest
        opfx.Mgabor  = 11;                % mask size
        opfx.show    = 0;                 % display results
        opfx.method  = 6;
    case {'gabor+','gaborfull'}
        opfx.d1      = 2;   % factor of downsampling along rows.
        opfx.d2      = 2;   % factor of downsampling along columns.
        opfx.u       = 4;   % number of scales
        opfx.v       = 4;   % number of orientations
        opfx.mm      = 7;  % rows of filter bank
        opfx.nn      = 7;  % columns of filter bank
        opfx.m       = numel(im)*opfx.u*opfx.v/opfx.d1/opfx.d2;
        opfx.show    = 0;
        opfx.method  = 11;
    case {'int','imtensity','src'}
        opfx.m = numel(im);
        opfx.method = 10;
    case {'sift'}
        opfx.m = 128;
        opfx.method = 13;
    case {'surf'}
        xs = detectSURFFeatures(im,'MetricThreshold',0.);
        opfx.m = 64;
        opfx.method = 14;
        opfx.xs = xs(1);
    case {'brisk'}
        xs = detectBRISKFeatures(im,'MinContrast',0.05);
        opfx.m = 64;
        opfx.method = 15;
        opfx.xs = xs(1);
    case {'hog'}
        opfx.m = 496;
        opfx.method = 15;
    case {'ggl','googlenet'}
        net = dagnn.DagNN.loadobj('nets/imagenet-googlenet-dag.mat') ;
        im_         = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
        im_         = im_ - net.meta.normalization.averageImage ;
        net.eval({'data', im_}) ;
        scores      = squeeze(gather(net.vars(end).value)) ;
        opfx.m      = numel(scores);
        opfx.net    = net;
        opfx.method = 9;
    case {'alx','alex','alexnet'}
        net         = load('nets/imagenet-caffe-alex.mat');
        imavg       = net.meta.normalization.averageImage;
        im_         = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
        if numel(imavg) == 3
            alx_avg = zeros(size(im_));
            alx_avg(:,:,1) = imavg(1);
            alx_avg(:,:,2) = imavg(2);
            alx_avg(:,:,3) = imavg(3);
        else
            alx_avg = imavg;
        end
        im_         = im_ - alx_avg ;
        res         = vl_simplenn(net, im_) ;
        scores      = squeeze(gather(res(end-2).x)) ;
        opfx.m      = numel(scores);
        opfx.net    = net;
        opfx.method = 8;
    case {'vgg1','vgg2','vgg3','vgg4'}
        v = feature(4);
        switch v
            case '1'
                net = load('nets/imagenet-vgg-f.mat');
            case '2'
                net = load('nets/imagenet-vgg-verydeep-16.mat');
            case '3'
                net = load('nets/imagenet-vgg-verydeep-19.mat');
            case '4'
                net = load('nets/imagenet-vgg-m-2048.mat');
        end
        net         = vl_simplenn_tidy(net);
        imavg       = net.meta.normalization.averageImage;
        im_         = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
        if numel(imavg) == 3
            vgg_avg = zeros(size(im_));
            vgg_avg(:,:,1) = imavg(1);
            vgg_avg(:,:,2) = imavg(2);
            vgg_avg(:,:,3) = imavg(3);
        else
            vgg_avg = imavg;
        end
        im_ = im_ - vgg_avg ;
        res = vl_simplenn(net, im_) ;
        scores = squeeze(gather(res(end-2).x)) ;
        opfx.m = numel(scores);
        opfx.net = net;
        opfx.method = 7;
    case {'xnet'}
        opfx.nlayer = 2;
        % net = load('/Users/domingomery/Dropbox/Mingo/Matlab/DP/data/x_cnn_500.mat') ;
        % net = load('/Users/domingomery/Dropbox/Mingo/Matlab/DP/dp_xnet/data/x_cnn.mat') ;
        net = load('x_cnn_500.mat') ;
        res       = vl_simplenn(net, im) ;
        scores    = squeeze(res(end-opfx.nlayer).x(:,:,:));
        opfx.m      = numel(scores);
        % imageMean = mean(imdb.images.data(:)) ;
        % Xim       = imdb.images.data - imageMean ;
        opfx.net    = net;
        opfx.method = 1;
        
end
