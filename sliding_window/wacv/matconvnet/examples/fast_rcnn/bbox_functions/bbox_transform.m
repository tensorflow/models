function targets = bbox_transform(ex_rois, gt_rois)
% Copyright (C) 2016 Hakan Bilen.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

ex_widths = ex_rois(:, 3) - ex_rois(:, 1) + 1.0 ;
ex_heights = ex_rois(:, 4) - ex_rois(:, 2) + 1.0 ;
ex_ctr_x = ex_rois(:, 1) + 0.5 * ex_widths ;
ex_ctr_y = ex_rois(:, 2) + 0.5 * ex_heights ;

gt_widths = gt_rois(:, 3) - gt_rois(:, 1) + 1.0 ;
gt_heights = gt_rois(:, 4) - gt_rois(:, 2) + 1.0 ;
gt_ctr_x = gt_rois(:, 1) + 0.5 * gt_widths ;
gt_ctr_y = gt_rois(:, 2) + 0.5 * gt_heights ;

targets_dx = (gt_ctr_x - ex_ctr_x) ./ ex_widths ;
targets_dy = (gt_ctr_y - ex_ctr_y) ./ ex_heights ;
targets_dw = log(gt_widths ./ ex_widths) ;
targets_dh = log(gt_heights ./ ex_heights) ;

targets = [targets_dx, targets_dy, targets_dw, targets_dh] ;