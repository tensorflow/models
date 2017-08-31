function imdb = add_bboxreg_targets(imdb)
% add bbox regression targets
%
% Copyright (C) 2016 Hakan Bilen.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

bgid = numel(imdb.classes.name) + 1;
imdb.boxes.ptarget = cell(numel(imdb.images.name),1);

count = 1;
% add targets
for i=1:numel(imdb.images.name)

  targets = zeros(numel(imdb.boxes.plabel{i}),4);
  pos = (imdb.boxes.plabel{i}>0 & imdb.boxes.plabel{i}<bgid);
  if isempty(pos)
    fprintf('no pos found (%d)\n',count);
    count = count + 1;
    assert(imdb.images.set(i)==3);
    continue;
  end

  ex_rois = imdb.boxes.pbox{i}(pos,:) ;
  gt_rois = imdb.boxes.gtbox{i}(imdb.boxes.pgtidx{i}(pos),:) ;

  targets(pos,:) = bbox_transform(ex_rois, gt_rois);

  imdb.boxes.ptarget{i} = targets;
end
ncls = numel(imdb.classes.name);

% compute means and stddevs
if ~isfield(imdb.boxes,'bboxMeanStd') || isempty(imdb.boxes.bboxMeanStd)

  sums = zeros(ncls,4);
  squared_sums = zeros(ncls,4);
  class_counts = zeros(ncls,1) + eps;

  for i=1:numel(imdb.boxes.ptarget)
    if imdb.images.set(i)<3
      pos =  (imdb.boxes.plabel{i}>0) & (imdb.boxes.plabel{i}<=ncls) ;
      labels = imdb.boxes.plabel{i}(pos);
      targets = imdb.boxes.ptarget{i}(pos,:);
      for c=1:ncls
        cls_inds = (labels==c);
        if sum(cls_inds)>0
          class_counts(c) = class_counts(c) + sum(cls_inds);
          sums(c,:) = sums(c,:) + sum(targets(cls_inds,:));
          squared_sums(c,:) = squared_sums(c,:) + sum(targets(cls_inds,:).^2);
        end
      end
    end
  end
  means = bsxfun(@rdivide,sums,class_counts);
  stds = sqrt(bsxfun(@rdivide,squared_sums,class_counts) - means.^2);

  imdb.boxes.bboxMeanStd{1} = means;
  imdb.boxes.bboxMeanStd{2} = stds;
  display('bbox target means:');
  display(means);
  display('bbox target stddevs:');
  display(stds);
else
  means = imdb.boxes.bboxMeanStd{1} ;
  stds = imdb.boxes.bboxMeanStd{2};
end

% normalize targets
for i=1:numel(imdb.boxes.ptarget)
%   if imdb.images.set(i)<3
    pos =  (imdb.boxes.plabel{i}>0) & (imdb.boxes.plabel{i}<=ncls) ;
    labels = imdb.boxes.plabel{i}(pos);
    targets = imdb.boxes.ptarget{i}(pos,:);
    for c=1:ncls
      cls_inds = (labels==c);
      if sum(cls_inds)>0
        targets(cls_inds,:) = bsxfun(@minus,targets(cls_inds,:),means(c,:));
        targets(cls_inds,:) = bsxfun(@rdivide,targets(cls_inds,:), stds(c,:));
      end
    end
    imdb.boxes.ptarget{i}(pos,:) = targets;
%   end
end
