function pred_boxes = bbox_transform_inv(boxes, deltas)
% Copyright (C) 2016 Hakan Bilen.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

if isempty(boxes), return; end

widths  = boxes(:,3) - boxes(:,1);
heights = boxes(:,4) - boxes(:,2);
ctr_x = boxes(:,1) + 0.5 * widths;
ctr_y = boxes(:,2) + 0.5 * heights;

dx = deltas(:,1);
dy = deltas(:,2);
dw = deltas(:,3);
dh = deltas(:,4);

pred_ctr_x = dx .* widths + ctr_x;
pred_ctr_y = dy .* heights + ctr_y;
pred_w = exp(dw) .* widths;
pred_h = exp(dh) .* heights;

pred_boxes = zeros(size(deltas), 'like', deltas);
% x1
pred_boxes(:, 1) = pred_ctr_x - 0.5 * pred_w;
% y1
pred_boxes(:, 2) = pred_ctr_y - 0.5 * pred_h;
% x2
pred_boxes(:, 3) = pred_ctr_x + 0.5 * pred_w;
% y2
pred_boxes(:, 4) = pred_ctr_y + 0.5 * pred_h;
