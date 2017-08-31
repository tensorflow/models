function cnn_toy_data_generator(dataDir)
%CNN_TOY_DATA_GENERATOR
%   Generates toy data in the given path: random image of triangles,
%   squares and circles.
%
%   The directory format is: '<dataDir>/<set>/<label>/<sample>.png', where
%   <set> is 'train' or 'val', <label> is an integer between 1 and 3, and
%   <sample> is the sample index.

% Copyright (C) 2017 Joao F. Henriques.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

  % Set random seed
  rng('default') ;
  rng(0) ;

  % The sets, and number of samples per label in each set
  sets = {'train', 'val'} ;
  numSamples = [1500, 150] ;

  % Number of polygon points in each class. The circle is created with 50
  % points.
  numPoints = [3, 4, 50] ;
  
  for s = 1:2  % Iterate sets
    for label = 1:3  % Iterate labels
      fprintf('Generating images for set %s, label %i...\n', sets{s}, label) ;
      
      mkdir(sprintf('%s/%s/%i', dataDir, sets{s}, label)) ;
      
      for i = 1:numSamples(s)  % Iterate samples
        % Points of a regular polygon, with random rotation and scale
        radius = randi([11, 14]) ;
        angles = rand(1) * 2 * pi + (0 : 2 * pi / numPoints(label) : 2 * pi) ;
        xs = 16.5 + cos(angles) * radius ;
        ys = 16.5 + sin(angles) * radius ;

        % Generate image
        image = poly2mask(xs, ys, 32, 32) ;
        
        % Save it
        imwrite(image, sprintf('%s/%s/%i/%04i.png', dataDir, sets{s}, label, i)) ;
      end
    end
  end

end

