function im = bbox_draw(im,boxes,c,t)

% copied from Ross Girshick
% Fast R-CNN
% Copyright (c) 2015 Microsoft
% Licensed under The MIT License [see LICENSE for details]
% Written by Ross Girshick
% --------------------------------------------------------
% source: https://github.com/rbgirshick/fast-rcnn/blob/master/matlab/showboxes.m
%
%
% Fast R-CNN
% 
% Copyright (c) Microsoft Corporation
% 
% All rights reserved.
% 
% MIT License
% 
% Permission is hereby granted, free of charge, to any person obtaining a
% copy of this software and associated documentation files (the "Software"),
% to deal in the Software without restriction, including without limitation
% the rights to use, copy, modify, merge, publish, distribute, sublicense,
% and/or sell copies of the Software, and to permit persons to whom the
% Software is furnished to do so, subject to the following conditions:
% 
% The above copyright notice and this permission notice shall be included
% in all copies or substantial portions of the Software.
% 
% THE SOFTWARE IS PROVIDED *AS IS*, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
% IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
% THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
% OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
% ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
% OTHER DEALINGS IN THE SOFTWARE.

image(im);
axis image;
axis off;
set(gcf, 'Color', 'white');

if nargin<3
  c = 'r';
  t = 2;
end

s = '-';
if ~isempty(boxes)
    x1 = boxes(:, 1);
    y1 = boxes(:, 2);
    x2 = boxes(:, 3);
    y2 = boxes(:, 4);
    line([x1 x1 x2 x2 x1]', [y1 y2 y2 y1 y1]', ...
        'color', c, 'linewidth', t, 'linestyle', s);
    for i = 1:size(boxes, 1)
        text(double(x1(i)), double(y1(i)) - 2, ...
            sprintf('%.4f', boxes(i, end)), ...
            'backgroundcolor', 'b', 'color', 'w', 'FontSize', 10);
    end
end
end