%VL_IMREADJPEG Load and transform images asynchronously
%   IMAGES = VL_IMREADJPEG(FILES) reads the specified cell array
%   FILES of JPEG files and returns at cell array of images IMAGES.
%
%   IMAGES = VL_IMREADJPEG(FILES, 'NumThreads', T) uses T parallel
%   threads to accelerate the operation. Note that this is independent
%   of the number of computational threads used by MATLAB.
%
%   VL_IMREADJPEG(FILES, 'Prefetch') starts reading the specified
%   images but returns immediately to MATLAB. Reading happens
%   concurrently with MATLAB in one or more separated threads.  A
%   subsequent call IMAGES=VL_IMREADJPEG(FILES) *specifying exactly
%   the same files in the same order* will then return the loaded
%   images. This can be sued to quickly load a batch of JPEG images as
%   MATLAB is busy doing something else.
%
%   The function can transforms the images on the fly in various
%   ways. Transformations are applied as follows:
%
%   1) An (H,W) image is loaded from disk.
%
%   2) A rectangular subset of the image is cropped and resized. The
%      geometry of the crop is determined as follows:
%
%      1) First, the shape (Ho,Wo) of the output image (i.e. the
%         resized crop) is determined. This, as determined by the
%         `Resize` option, can be either the same as the input image
%         (H,W) or one or both of height and width can be set to an
%         arbitrary value.
%
%      2) Given the output shape (Ho,Wo) of the crop, the shape
%         (Hi,Wi) and location of the crop in the input image is
%         determined.  First, an anisotropy ratio (change in aspect
%         ratio) is selected according to `CropAnisotropy`. Given
%         that, the input crop rectangle is scaled to fill a certain
%         percentage of the input image according to `CropSize`.
%         Finally, the crop is extracted either from the middle of the
%         input image or at a random location according to
%         `CropLocation`.
%
%   3) The cropped and resized image undergoes color post
%      post-processing, including mean subtraction
%      (`SubtractAverage`), random color shift (`Brightness`), and
%      random changes in saturation (`Saturation`), and contrast
%      (`Contrast`).
%
%   The function takes the following options:
%
%   `Prefetch`:: not specified
%     If specified, run without blocking (see above).
%
%   `Verbose`:: not specified
%     If specified, increase the verbosity level.
%
%   `Pack`:: not specified
%     If specified, pack all K images in a single H x W x 3 x K
%     array. This requires using the option `'Resize',[H W]` in order
%     to set the output size to be the same for all images. Note
%     furthermore that gray-scale images are automatically extended to
%     three color channels.
%
%   `GPU`:: not specified
%     If specified, return GPU arrays instead of standard arrays. Note
%     that, using the `Prefetch` option, the data is copied to the GPU
%     in parallel with computations.
%
%   `NumThreads`:: `1`
%     Specify the number of threads used to read images. This number
%     must be at least 1. Note that it does not make sense to specify
%     a number larger than the number of available CPU cores, and
%     often fewer threads are sufficient as reading images is memory
%     access bound rather than CPU bound.
%
%   `Resize`:: not specified
%     If specified, turn on image resizing. The argument can either
%     specify the desired height and width `[H, W]` or be simply a
%     scalar `S`. In the latter case, the image is resized
%     isotropically so that the shorter side is equal to `S`.
%
%     Resizing uses bilinear interpolation. When shrinking, a simple
%     form of antialiasing is used by stretching the bilinear filter
%     over several input pixels to average them. The method is the
%     same as MATLAB `imresize` function (the two functions are
%     numerically equivalent).
%
%   `CropAnisotropy`:: `[1 1]`
%     Specify the minimum and maximum value of the anisotropy ratio
%     for the selected crops. The default value [1 1] means that the
%     crops are extracted isotropically, i.e. with the same amount of
%     stretch along horizontal and vertical directions. A value of `[0
%     0]` stretches the crop to fit the input image. A value `[A1 A2]`
%     means that an anisotropy ratio A = Wcrop/Hcrop is sampled
%     uniformly at random for each image from the specified interval.
%
%   `CropSize`:: `[1 1]`
%     Specifies the maximum and minimum size of a crop, as a
%     percentage of the maximum size it can take to fill the input
%     image (the latter is determined *after* fixing the
%     anisotropy). Using `[1 1]` means that the crop is always sized
%     to fill up as much area as possible.
%
%   `CropLocation`:: `'center'`
%     If set to `'random'`, the crop location in the input image is
%     sampled uniformly at random among all possible shifts such that
%     the crop is fully contained in the image.
%
%   `Flip`:: not specified
%     If specified, randomly flips a crop horizontally with 50%
%     probability.
%
%   `SubtractAverage`:: `[0 0 0]`
%     Subtract the specified RGB triplet from each pixel.
%
%   `Brightness`:: `zeros(3)`
%     Standard deviation `B` of the brightness shift. Each image is
%     modified by adding the color shift `B*w`, where `w` is a 3D
%     vector sampled from the standard Normal distribution.
%
%   `Contrast`:: `0`
%     Deviation `C` of the contrast shift. It must be between 0 and
%     1. A value of 0 amounts to no contrast shift. Otherwise, a
%     contrast shift is sampled at random, using `C` as an upper
%     bound.
%
%   `Saturation`:: `0`
%     The same as `Contrast`, but for saturation.
%
%   `Interpolation`:: `'bilinear'`
%     The interpolation method; one of `box`, `bilinear`, `bicubic`,
%     `lanczos2`, and `lanczos3`. The function uses interpolators
%     equivalent to MATLAB's `imresize`.
%
%   Further details on the processing performed by the function can be
%   found in the PDF manual.

% Copyright (C) 2014-16 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).
