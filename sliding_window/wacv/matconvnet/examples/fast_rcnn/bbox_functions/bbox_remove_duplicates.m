function bboxeso = bbox_remove_duplicates(bboxes, minSize, maxNum)
% Copyright (C) 2016 Hakan Bilen.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).
bboxeso = cell(size(bboxes));
for i=1:numel(bboxes)
  bbox = bboxes{i};
  % remove small bbox
  isGood = (bbox(:,3)>=bbox(:,1)-1+minSize) & (bbox(:,4)>=bbox(:,2)-1+minSize);
  bbox = bbox(isGood,:);
  % remove duplicate ones
  [dummy, uniqueIdx] = unique(bbox, 'rows', 'first');
  uniqueIdx = sort(uniqueIdx);
  bbox = bbox(uniqueIdx,:);
  % limit number for training
  nB = min(size(bbox,1),maxNum);

  bboxeso{i} = bbox(1:nB,:);
end
