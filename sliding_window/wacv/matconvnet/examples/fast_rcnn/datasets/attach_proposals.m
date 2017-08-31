function imdb = attach_proposals(imdb,boxes,fgThresh,bgThreshHi,bgThreshLo)
%
% Copyright (C) 2016 Hakan Bilen.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

% gtboxes = imdb.images.gtboxes;
% imdb.images = rmfield(imdb.images,'gtboxes');

if numel(boxes)~=numel(imdb.images.name)
  error('Wrong number of boxes');
end

minSize = 20;
maxNum = inf;
visualize = false ;

boxes = bbox_remove_duplicates(boxes, minSize, maxNum);

%% Get obj proposals.
imdb.boxes.pbox   = cell(numel(imdb.images.name),1);
imdb.boxes.plabel = cell(numel(imdb.images.name),1);
imdb.boxes.piou   = cell(numel(imdb.images.name),1);
imdb.boxes.pgtidx = cell(numel(imdb.images.name),1);

bglabel = numel(imdb.classes.name) + 1;

for i=1:numel(boxes)
  gtbb = imdb.boxes.gtbox{i};
  gtlabels = imdb.boxes.gtlabel{i};

  if imdb.images.set(i)<3
    boxes{i} = [boxes{i};gtbb];
  end

  pbox = boxes{i};

  if size(pbox,2)~=4
    error('wrong box dim');
  end

  plabel = zeros(size(boxes{i},1),1);
  pgtidx = zeros(size(boxes{i},1),1);


  iou = bbox_overlap(single(pbox),single(gtbb));
  [max_iou,gt_assignments] = max(iou,[],2);
  if imdb.images.set(i)<3 && all(max_iou<1)
    error('Ground truth boxes are not added!');
  end

  piou = max_iou;

  max_labels = gtlabels(gt_assignments);

  % positive and negative boxes
  pos = (max_iou>=fgThresh);
  neg = (max_iou>=bgThreshLo & max_iou<bgThreshHi);

  if visualize
    [so,si] = max(max_iou);
    im = imread([imdb.imageDir, filesep, imdb.images.name{i}]);
    [h,w,~] = size(im);
    assert(all(pbox(:,3)<=w) && all(pbox(:,4)<=h));
    for j=1:size(pbox,1)
      if max_iou(j)>=0.5
        bbox_draw(im,pbox(j,:),'g',1);
      elseif max_iou(j)>=0.1 &&  max_iou(j)<0.5
        bbox_draw(im,pbox(j,:),'r',1);
      else
        bbox_draw(im,pbox(j,:),'b',1);
      end
      pause;
    end
  end

  % Assign labels to box proposals.
  plabel(pos) = max_labels(pos);
  pgtidx(pos) = gt_assignments(pos);
  plabel(neg) = bglabel;

  if imdb.images.set(i)~=3
    keep = plabel>0;
    plabel = plabel(keep);
    pgtidx = pgtidx(keep);
    pbox = pbox(keep,:);
    piou = piou(keep);
  end

  imdb.boxes.pbox{i} = pbox;
  imdb.boxes.plabel{i} = plabel;
  imdb.boxes.pgtidx{i} = pgtidx;
  imdb.boxes.piou{i} = piou;
end
