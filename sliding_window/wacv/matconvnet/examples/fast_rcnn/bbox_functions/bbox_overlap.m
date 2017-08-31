function overlaps = bbox_overlap(boxes1,boxes2)
% Copyright (C) 2016 Hakan Bilen.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).
x11 = boxes1(:,1);
y11 = boxes1(:,2);
x12 = boxes1(:,3);
y12 = boxes1(:,4);

x21 = boxes2(:,1);
y21 = boxes2(:,2);
x22 = boxes2(:,3);
y22 = boxes2(:,4);

N1 = size(boxes1,1);
N2 = size(boxes2,1);

area1 = (x12-x11+1) .* (y12-y11+1);
area2 = (x22-x21+1) .* (y22-y21+1);

overlaps = zeros(N1,N2);

for i=1:N1

  xx1 = max(x11(i), x21);
  yy1 = max(y11(i), y21);
  xx2 = min(x12(i), x22);
  yy2 = min(y12(i), y22);

  w = max(0.0, xx2-xx1+1);
  h = max(0.0, yy2-yy1+1);

  inter = w.*h;
  overlaps(i,:) = inter ./ (area1(i) + area2 - inter);
end

