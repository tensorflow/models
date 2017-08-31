function boxOut = bbox_scale2(boxIn,scale,szOut)
% Copyright (C) 2016 Hakan Bilen.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

if isempty(boxIn), boxOut = []; return; end

boxOut = scale * (boxIn-1) + 1;

boxOut = [max(1,round(boxOut(:,1))),...
  max(1,round(boxOut(:,2))),...
  min(szOut(1),round(boxOut(:,3))),...
  min(szOut(2),round(boxOut(:,4)))];
