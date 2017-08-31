function x = wacv_fx(im,opfx)

switch opfx.method
    case 1 % dp
        im      = im - opfx.imageMean;
        res     = vl_simplenn(opfx.net, im) ;
        scores  = squeeze(res(end-opfx.nlayer).x(:,:,:));
        x = scores(:)';
    case 2 % lbp / lbpri / slbp
        x = Bfx_lbp(im,opfx);
    case 3 % bsif
        x = Bfx_bsif(im,opfx);
    case 4 % txh haralick
        x = Bfx_haralick(im*255,opfx);
    case 5 % clp
        x = Bfx_clp(im,opfx);
    case 6 % gabor
        x = Bfx_gabor(im,opfx);
    case 11 % gabor+
        x = Bfx_gaborfull(im,opfx);
    case 7 % vgg
        net           = opfx.net;
        [N,M]         = size(im);
        im3           = zeros(N,M,3);
        im3(:,:,1)    = im;
        im3(:,:,2)    = im;
        im3(:,:,3)    = im;
        imavg         = net.meta.normalization.averageImage;
        im_           = single(im3) ; % note: 255 range
        im_           = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
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
        x = scores(:)';
        
    case 8 % alex-net
        net           = opfx.net;
        [N,M]         = size(im);
        im3           = zeros(N,M,3);
        im3(:,:,1)    = im;
        im3(:,:,2)    = im;
        im3(:,:,3)    = im;
        imavg         = net.meta.normalization.averageImage;
        im_           = single(im3) ;
        im_           = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
        if numel(imavg) == 3
            alx_avg = zeros(size(im_));
            alx_avg(:,:,1) = imavg(1);
            alx_avg(:,:,2) = imavg(2);
            alx_avg(:,:,3) = imavg(3);
        else
            alx_avg = imavg;
        end
        im_ = im_ - alx_avg ;
        res = vl_simplenn(net, im_) ;
        scores = squeeze(gather(res(end-2).x)) ;
        x = scores(:)';
    case 9 % google-net
        net           = opfx.net;
        [N,M]         = size(im);
        im3           = zeros(N,M,3);
        im3(:,:,1)    = im;
        im3(:,:,2)    = im;
        im3(:,:,3)    = im;
        im_           = single(im3) ;
        im_           = imresize(im_, net.meta.normalization.imageSize(1:2)) ;
        im_           = im_ - net.meta.normalization.averageImage ;
        net.eval({'data', im_}) ;
        scores        = squeeze(gather(net.vars(end).value)) ;
        op.m          = numel(scores);
        x             = scores(:)';
    case 10 % int
        x = (im(:)/norm(im(:)))';
    case 12 % dft
        [N,M]         = size(im);
        N2            = round(N/2);
        M2            = round(M/2);
        xf            = fft2(im);
        xf2           = abs(xf(1:N2,1:M2));
        x             = xf2(:)';
    case 17 % dct
        xf            = dct2(im);
        x             = xf(:)';
    case 13 % sift
        [~,xs]        = vl_sift(single(im),'frames',[16 16 4 0]');
        x             = xs(:)';
    case 14 % surf
        xx = opfx.xs;xx.Scale = 4;xx.SignOfLaplacian = -1;xx.Orientation=0;xx.Location= [16 16];
        xf            = extractFeatures(im, xx);
        x             = xf(1,:);
    case 15 % hog
        xs            = vl_hog(single(im),8);
        x             = xs(:)';
    case 16 % brisk
        xx = opfx.xs;xx.Scale = 4;xx.SignOfLaplacian = -1;xx.Orientation=0;xx.Location= [16 16];
        xf            = extractFeatures(im, xx);
        x             = xf(1,:);
end
