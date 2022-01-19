import tensorflow as tf

def get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W,)
    - y: flattened tensor of shape (B*H*W,)
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)

    return tf.gather_nd(img, indices)


def bilinear_sampler(img, grid, align_corners=False):
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.
    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.
    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.
    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    """
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    max_y = tf.cast(H, 'int32')
    max_x = tf.cast(W, 'int32')
    zero = tf.zeros([], dtype='int32')

    x = grid[:,0,:,:]
    y = grid[:,1,:,:]

    # rescale x and y to [0, W-1/H-1]
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')

    if align_corners:
      x = ((x + 1.0) / 2) * tf.cast(max_x-1, 'float32')
      y = ((y + 1.0) / 2) * tf.cast(max_y-1, 'float32')
    else:
      x = ((x + 1.0) * tf.cast(max_x, 'float32') - 1.0) / 2.0
      y = ((y + 1.0) * tf.cast(max_y, 'float32') - 1.0) / 2.0

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # calculate deltas
    wa = (tf.cast(x1, 'float32')-x) * (tf.cast(y1, 'float32')-y)
    wb = (tf.cast(x1, 'float32')-x) * (y-tf.cast(y0, 'float32'))
    wc = (x-tf.cast(x0, 'float32')) * (tf.cast(y1, 'float32')-y)
    wd = (x-tf.cast(x0, 'float32')) * (y-tf.cast(y0, 'float32'))

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # clip to range [0, H-1/W-1] to not violate img boundaries
    
    x0 = tf.clip_by_value(x0, zero, max_x - 1)
    x1 = tf.clip_by_value(x1, zero, max_x - 1)
    y0 = tf.clip_by_value(y0, zero, max_y - 1)
    y1 = tf.clip_by_value(y1, zero, max_y - 1)
    
    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')


    # compute output
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])

    return out

def vert_align(feats, verts, return_packed: bool = False, align_corners: bool = True):
  if tf.is_tensor(verts):
    if len(verts.shape) != 3:
      raise ValueError("verts tensor should be 3 dimensional")
    grid = verts
  else:
    raise ValueError(
        "verts must be a tensor.")

  grid = grid[:, None, :, :2]

  if tf.is_tensor(feats):
    feats = [feats]
  for feat in feats:
    if len(feat.shape) != 4:
      raise ValueError("feats.shape (N, C, H, W)")
    if grid.shape[0] != feat.shape[0]:
      raise ValueError("inconsistent batch dimension")

  feats_sampled = []
  for feat in feats:
    feat_sampled = tf.transpose(
        bilinear_sampler(tf.transpose(feat, [0, 2, 3, 1]), tf.transpose(grid, [0, 3, 1, 2]), align_corners=align_corners)
        ,[0, 3, 1, 2])
    feat_sampled = tf.transpose(tf.squeeze(feat_sampled, axis = 2), [0,2,1])
    feats_sampled.append(feat_sampled)
  feats_sampled = tf.concat(feats_sampled, axis = 2)

  return feats_sampled
  