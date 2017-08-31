function boxes = bbox_clip(boxes, im_size)
% bbox_clip Clip boxes to image boundaries.
% 
% Copyright (C) 2016 Hakan Bilen.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).
boxes(:,1) = max(min(boxes(:,1),im_size(2)),1);
boxes(:,2) = max(min(boxes(:,2),im_size(1)),1);
boxes(:,3) = max(min(boxes(:,3),im_size(2)),1);
boxes(:,4) = max(min(boxes(:,4),im_size(1)),1);
