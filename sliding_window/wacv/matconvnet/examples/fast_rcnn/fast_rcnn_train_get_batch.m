function [imo,rois,labels,targets] = fast_rcnn_train_get_batch(images, imdb, batch, opts)
% FAST_RCNN_GET_BATCH_TRAIN  Generates mini-batches for Fast-RCNN train

% opts.numFgRoisPerImg = 128;
% opts.numRoisPerImg = 64;
% opts.maxScale = 1000;
% opts.bgLabel = 21;
% opts.visualize = 0;
% opts.scale = 600;
% opts.interpolation = 'bicubic';
% opts.averageImage = [];
% opts.numThreads = 2;
% opts.prefetch = true;
%
% Copyright (C) 2016 Hakan Bilen.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

if isempty(images)
  imo = [] ;
  rois = [] ;
  labels = [] ;
  targets = [] ;
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
  labels = [] ;
  targets = [] ;
  return ;
end

if fetch
  ims = vl_imreadjpeg(images,'numThreads',opts.numThreads) ;
else
  ims = images ;
end

maxW = 0;
maxH = 0;

% labels = imdb.images.label(:,batch);

pboxes   = cell(1,numel(batch));
plabels  = cell(1,numel(batch));
ptargets = cell(1,numel(batch));

% get fg and bg rois
for b=1:numel(batch)
  pbox   = imdb.boxes.pbox{batch(b)};
  plabel = imdb.boxes.plabel{batch(b)};
  ptarget = imdb.boxes.ptarget{batch(b)};

  if size(pbox,2)~=4
    error('wrong box size');
  end

  % get pos boxes
  pos = find((plabel~=opts.bgLabel) & (plabel > 0)) ;
  npos = numel(pos);
  % get neg boxes
  neg = find((plabel==opts.bgLabel)) ;
  nneg = numel(neg);

    bbox = [];
    label = [];
    target = [];

    opts.numFgRoisPerImg = min(npos,opts.numFgRoisPerImg);
    nBneg = min(nneg,opts.numRoisPerImg-opts.numFgRoisPerImg);

    if npos>0
      r = randperm(npos);
      p = pos(r(1:opts.numFgRoisPerImg));
      bbox = pbox(p,:);
      label = plabel(p);
      target = ptarget(p,:);
    end
    if nneg>0
      r = randperm(nneg);

      n = neg(r(1:nBneg));
      bbox = [bbox ; pbox(n,:)];
      label = [label ; plabel(n)];
      target = [target ; ones(size(ptarget(n,:)))];
    end
  pboxes{b} = bbox;
  plabels{b} = label;
  ptargets{b} = target;
end

if isempty(pboxes)
  warning('No gt box\n');
end

labels = vertcat(plabels{:});
targets = vertcat(ptargets{:});

% rescale images and rois
rois = [];
imre = cell(1,numel(batch));
for b=1:numel(batch)
  imSize = size(ims{b});

  h = imSize(1);
  w = imSize(2);

  factor = max(opts.scale(1)/h,opts.scale(1)/w);

  if any([h*factor,w*factor]>opts.maxScale)
    factor = min(opts.maxScale/h,opts.maxScale/w);
  end

  if abs(factor-1)>1e-3
    imre{b} = imresize(ims{b},factor,'Method',opts.interpolation);
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

  rois = [rois [b*ones(1,nB) ; tbbox' ] ];
end

imo = zeros(maxH,maxW,size(imre{1},3),numel(batch),'single');
for b=1:numel(batch)
  % subtract mean
  if ~isempty(opts.averageImage)
    imre{b} = single(bsxfun(@minus,imre{b},opts.averageImage));
  end
  sz = size(imre{b});
  imo(1:sz(1),1:sz(2),:,b) = single(imre{b});
end
