function vl_tshow(T, varargin)
%VL_TSHOW Visualize a 4D tensor.
%   VL_TSHOW(T) shows the 4D tensor T in the current figure.
%
%   The tensor is shown as a montage of 2D slices (e.g. filters), with the
%   3rd dimension stacked along the rows and the 4th dimension along the
%   columns.
%
%   VL_TSHOW(T, 'option', value, ...) accepts the following options:
%
%   `labels`:: true
%     If true, labels the x/y axis of the montage.
%
%   Any additional options are passed to IMAGESC (e.g. to set the parent
%   axes, or other properties).

% Copyright (C) 2017 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.labels = true ;
[opts, varargin] = vl_argparse(opts, varargin, 'nonrecursive') ;

assert((isnumeric(T) || islogical(T)) && ndims(T) <= 4, ...
  'T must be a 4D numeric or logical tensor.') ;

% Stack input channels along rows (merge 1st dim. with 3rd), and output
% channels along columns (merge 2nd dim. with 4th), to form a 2D image
sz = size(T) ;
sz(end+1:4) = 1 ;
T = reshape(permute(T, [1 3 2 4]), sz(1) * sz(3), sz(2) * sz(4)) ;

% Display it
h = imagesc(T, varargin{:}) ;

ax = get(h, 'Parent') ;
axis(ax, 'image') ;

% Display grid between filters
set(ax, 'XGrid', 'on', 'YGrid', 'on', 'GridAlpha', 1, ...
        'TickLength', [0 0], 'XTickLabel', {}, 'YTickLabel', {}, ...
        'YTick', sz(1) + 0.5 : sz(1) : sz(1) * sz(3) - 0.5, ...
        'XTick', sz(2) + 0.5 : sz(2) : sz(2) * sz(4) - 0.5) ;

if opts.labels
  xlabel(sprintf('Output channels (%i)', sz(4)), 'Parent', ax) ;
  ylabel(sprintf('Input channels (%i)', sz(3)), 'Parent', ax) ;
end

