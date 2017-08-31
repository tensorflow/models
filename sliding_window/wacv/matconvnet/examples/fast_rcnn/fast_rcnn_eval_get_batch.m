function [imo,rois] = fast_rcnn_eval_get_batch(images, imdb, batch, opts)
% FAST_RCNN_GET_BATCH_EVAL Load, preprocess, and pack images for CNN
% evaluation

% opts.numFgRoisPerImg = 128;
% opts.numRoisPerImg = 64;
% opts.maxScale = 1000;
% opts.bgLabel = 21;
% opts.visualize = 1;
% opts.scale = 600;
% opts.interpolation = 'bicubic';
% opts.averageImage = [];
% opts.numThreads = 2;
%
% Copyright (C) 2016 Hakan Bilen.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

if isempty(images)
  imo = [] ;
  rois = [] ;
  return ;
end

% fetch is true if images is a list of filenames (instead of
% a cell array of images)
fetch = ischar(images{1}) ;

% prefetch is used to load images in a separate thread
prefetch = fetch & opts.prefetch ;

if prefetch
  vl_imreadjpeg(images, 'numThreads',opts.numThreads,'prefetch') ;
  imo = [] ;
  rois = [] ;
  return ;
end

if fetch
  ims = vl_imreadjpeg(images,'numThreads',opts.numThreads) ;
else
  ims = images ;
end



imre = cell(1,numel(batch));
maxW = 0;
maxH = 0;

pboxes  = cell(1,numel(batch));

% get fg and bg rois
for b=1:numel(batch)
  pbox   = imdb.boxes.pbox{batch(b)};

  if size(pbox,2)~=4
    error('wrong box size');
  end

  pboxes{b} = pbox;
end

% rescale images and rois
rois = [];
for b=1:numel(batch)
  imSize = size(ims{b});

  h = imSize(1);
  w = imSize(2);

  factor = max(opts.scale(1)/h,opts.scale(1)/w);

  if any([h*factor,w*factor]>opts.maxScale)
    factor = min(opts.maxScale/h,opts.maxScale/w);
  end

  if abs(factor-1)>1e-3
    imre{b} = imresize(ims{b},factor,'Method',opts.interpolation,...
        'antialiasing', false);
  else
    imre{b} = ims{b};
  end

  if imdb.boxes.flip(batch(b))
    im = imre{b};
    imre{b} = im(:,end:-1:1,:);
  end

  imreSize = size(imre{b});

  maxH = max(imreSize(1),maxH);
  maxW = max(imreSize(2),maxW);

  % adapt bounding boxes into new coord
  bbox = pboxes{b};
  if any(bbox(:)<=0)
    error('bbox error');
  end

  nB = size(bbox,1);
  tbbox = bbox_scale(bbox,factor,[imreSize(2) imreSize(1)]);
  if any(tbbox(:)<=0)
    error('tbbox error');
  end

  rois = [rois [b*ones(1,nB); tbbox' ] ];
end

% rois = single(rois);
imo = zeros(maxH,maxW,size(imre{1},3),numel(batch),'single');
for b=1:numel(batch)
  if ~isempty(opts.averageImage)
    imre{b} = single(bsxfun(@minus,imre{b},opts.averageImage));
  end
  sz = size(imre{b});
  imo(1:sz(1),1:sz(2),:,b) = single(imre{b});
end



