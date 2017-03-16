"""TensorFlow ops for directed graphs."""

import tensorflow as tf

from syntaxnet.util import check


def ArcPotentialsFromTokens(source_tokens, target_tokens, weights):
  r"""Returns arc potentials computed from token activations and weights.

  For each batch of source and target token activations, computes a scalar
  potential for each arc as the 3-way product between the activation vectors of
  the source and target of the arc and the |weights|.  Specifically,

    arc[b,s,t] =
        \sum_{i,j} source_tokens[b,s,i] * weights[i,j] * target_tokens[b,t,j]

  Note that the token activations can be extended with bias terms to implement a
  "biaffine" model (Dozat and Manning, 2017).

  Args:
    source_tokens: [B,N,S] tensor of batched activations for the source token in
                   each arc.
    target_tokens: [B,N,T] tensor of batched activations for the target token in
                   each arc.
    weights: [S,T] matrix of weights.

    B,N may be statically-unknown, but S,T must be statically-known.  The dtype
    of all arguments must be compatible.

  Returns:
    [B,N,N] tensor A of arc potentials where A_{b,s,t} is the potential of the
    arc from s to t in batch element b.  The dtype of A is the same as that of
    the arguments.  Note that the diagonal entries (i.e., where s==t) represent
    self-loops and may not be meaningful.
  """
  # All arguments must have statically-known rank.
  check.Eq(source_tokens.get_shape().ndims, 3, 'source_tokens must be rank 3')
  check.Eq(target_tokens.get_shape().ndims, 3, 'target_tokens must be rank 3')
  check.Eq(weights.get_shape().ndims, 2, 'weights must be a matrix')

  # All activation dimensions must be statically-known.
  num_source_activations = weights.get_shape().as_list()[0]
  num_target_activations = weights.get_shape().as_list()[1]
  check.NotNone(num_source_activations, 'unknown source activation dimension')
  check.NotNone(num_target_activations, 'unknown target activation dimension')
  check.Eq(source_tokens.get_shape().as_list()[2], num_source_activations,
           'dimension mismatch between weights and source_tokens')
  check.Eq(target_tokens.get_shape().as_list()[2], num_target_activations,
           'dimension mismatch between weights and target_tokens')

  # All arguments must share the same type.
  check.Same([weights.dtype.base_dtype,
              source_tokens.dtype.base_dtype,
              target_tokens.dtype.base_dtype],
             'dtype mismatch')

  source_tokens_shape = tf.shape(source_tokens)
  target_tokens_shape = tf.shape(target_tokens)
  batch_size = source_tokens_shape[0]
  num_tokens = source_tokens_shape[1]
  with tf.control_dependencies([
      tf.assert_equal(batch_size, target_tokens_shape[0]),
      tf.assert_equal(num_tokens, target_tokens_shape[1])]):
    # Flatten out the batch dimension so we can use one big multiplication.
    targets_bnxt = tf.reshape(target_tokens, [-1, num_target_activations])

    # Matrices are row-major, so we arrange for the RHS argument of each matmul
    # to have its transpose flag set.  That way no copying is required to align
    # the rows of the LHS with the columns of the RHS.
    weights_targets_bnxs = tf.matmul(targets_bnxt, weights, transpose_b=True)

    # The next computation is over pairs of tokens within each batch element, so
    # restore the batch dimension.
    weights_targets_bxnxs = tf.reshape(
        weights_targets_bnxs, [batch_size, num_tokens, num_source_activations])

    # Note that this multiplication is repeated across the batch dimension,
    # instead of being one big multiplication as in the first matmul.  There
    # doesn't seem to be a way to arrange this as a single multiplication given
    # the pairwise nature of this computation.
    arcs_bxnxn = tf.matmul(source_tokens, weights_targets_bxnxs,
                           transpose_b=True)
    return arcs_bxnxn


def ArcSourcePotentialsFromTokens(tokens, weights):
  r"""Returns arc source potentials computed from tokens and weights.

  For each batch of token activations, computes a scalar potential for each arc
  as the product between the activations of the source token and the |weights|.
  Specifically,

    arc[b,s,:] = \sum_{i} weights[i] * tokens[b,s,i]

  Args:
    tokens: [B,N,S] tensor of batched activations for source tokens.
    weights: [S] vector of weights.

    B,N may be statically-unknown, but S must be statically-known.  The dtype of
    all arguments must be compatible.

  Returns:
    [B,N,N] tensor A of arc potentials as defined above.  The dtype of A is the
    same as that of the arguments.  Note that the diagonal entries (i.e., where
    s==t) represent self-loops and may not be meaningful.
  """
  # All arguments must have statically-known rank.
  check.Eq(tokens.get_shape().ndims, 3, 'tokens must be rank 3')
  check.Eq(weights.get_shape().ndims, 1, 'weights must be a vector')

  # All activation dimensions must be statically-known.
  num_source_activations = weights.get_shape().as_list()[0]
  check.NotNone(num_source_activations, 'unknown source activation dimension')
  check.Eq(tokens.get_shape().as_list()[2], num_source_activations,
           'dimension mismatch between weights and tokens')

  # All arguments must share the same type.
  check.Same([weights.dtype.base_dtype,
              tokens.dtype.base_dtype],
             'dtype mismatch')

  tokens_shape = tf.shape(tokens)
  batch_size = tokens_shape[0]
  num_tokens = tokens_shape[1]

  # Flatten out the batch dimension so we can use a couple big matmuls.
  tokens_bnxs = tf.reshape(tokens, [-1, num_source_activations])
  weights_sx1 = tf.expand_dims(weights, 1)
  sources_bnx1 = tf.matmul(tokens_bnxs, weights_sx1)
  sources_bnxn = tf.tile(sources_bnx1, [1, num_tokens])

  # Restore the batch dimension in the output.
  sources_bxnxn = tf.reshape(sources_bnxn, [batch_size, num_tokens, num_tokens])
  return sources_bxnxn


def RootPotentialsFromTokens(root, tokens, weights):
  r"""Returns root selection potentials computed from tokens and weights.

  For each batch of token activations, computes a scalar potential for each root
  selection as the 3-way product between the activations of the artificial root
  token, the token activations, and the |weights|.  Specifically,

    roots[b,r] = \sum_{i,j} root[i] * weights[i,j] * tokens[b,r,j]

  Args:
    root: [S] vector of activations for the artificial root token.
    tokens: [B,N,T] tensor of batched activations for root tokens.
    weights: [S,T] matrix of weights.

    B,N may be statically-unknown, but S,T must be statically-known.  The dtype
    of all arguments must be compatible.

  Returns:
    [B,N] matrix R of root-selection potentials as defined above.  The dtype of
    R is the same as that of the arguments.
  """
  # All arguments must have statically-known rank.
  check.Eq(root.get_shape().ndims, 1, 'root must be a vector')
  check.Eq(tokens.get_shape().ndims, 3, 'tokens must be rank 3')
  check.Eq(weights.get_shape().ndims, 2, 'weights must be a matrix')

  # All activation dimensions must be statically-known.
  num_source_activations = weights.get_shape().as_list()[0]
  num_target_activations = weights.get_shape().as_list()[1]
  check.NotNone(num_source_activations, 'unknown source activation dimension')
  check.NotNone(num_target_activations, 'unknown target activation dimension')
  check.Eq(root.get_shape().as_list()[0], num_source_activations,
           'dimension mismatch between weights and root')
  check.Eq(tokens.get_shape().as_list()[2], num_target_activations,
           'dimension mismatch between weights and tokens')

  # All arguments must share the same type.
  check.Same([weights.dtype.base_dtype,
              root.dtype.base_dtype,
              tokens.dtype.base_dtype],
             'dtype mismatch')

  root_1xs = tf.expand_dims(root, 0)

  tokens_shape = tf.shape(tokens)
  batch_size = tokens_shape[0]
  num_tokens = tokens_shape[1]

  # Flatten out the batch dimension so we can use a couple big matmuls.
  tokens_bnxt = tf.reshape(tokens, [-1, num_target_activations])
  weights_targets_bnxs = tf.matmul(tokens_bnxt, weights, transpose_b=True)
  roots_1xbn = tf.matmul(root_1xs, weights_targets_bnxs, transpose_b=True)

  # Restore the batch dimension in the output.
  roots_bxn = tf.reshape(roots_1xbn, [batch_size, num_tokens])
  return roots_bxn


def CombineArcAndRootPotentials(arcs, roots):
  """Combines arc and root potentials into a single set of potentials.

  Args:
    arcs: [B,N,N] tensor of batched arc potentials.
    roots: [B,N] matrix of batched root potentials.

  Returns:
    [B,N,N] tensor P of combined potentials where
      P_{b,s,t} = s == t ? roots[b,t] : arcs[b,s,t]
  """
  # All arguments must have statically-known rank.
  check.Eq(arcs.get_shape().ndims, 3, 'arcs must be rank 3')
  check.Eq(roots.get_shape().ndims, 2, 'roots must be a matrix')

  # All arguments must share the same type.
  dtype = arcs.dtype.base_dtype
  check.Same([dtype, roots.dtype.base_dtype], 'dtype mismatch')

  roots_shape = tf.shape(roots)
  arcs_shape = tf.shape(arcs)
  batch_size = roots_shape[0]
  num_tokens = roots_shape[1]
  with tf.control_dependencies([
      tf.assert_equal(batch_size, arcs_shape[0]),
      tf.assert_equal(num_tokens, arcs_shape[1]),
      tf.assert_equal(num_tokens, arcs_shape[2])]):
    return tf.matrix_set_diag(arcs, roots)


def LabelPotentialsFromTokens(tokens, weights):
  r"""Computes label potentials from tokens and weights.

  For each batch of token activations, computes a scalar potential for each
  label as the product between the activations of the source token and the
  |weights|.  Specifically,

    labels[b,t,l] = \sum_{i} weights[l,i] * tokens[b,t,i]

  Args:
    tokens: [B,N,T] tensor of batched token activations.
    weights: [L,T] matrix of weights.

    B,N may be dynamic, but L,T must be static.  The dtype of all arguments must
    be compatible.

  Returns:
    [B,N,L] tensor of label potentials as defined above, with the same dtype as
    the arguments.
  """
  check.Eq(tokens.get_shape().ndims, 3, 'tokens must be rank 3')
  check.Eq(weights.get_shape().ndims, 2, 'weights must be a matrix')

  num_labels = weights.get_shape().as_list()[0]
  num_activations = weights.get_shape().as_list()[1]
  check.NotNone(num_labels, 'unknown number of labels')
  check.NotNone(num_activations, 'unknown activation dimension')
  check.Eq(tokens.get_shape().as_list()[2], num_activations,
           'activation mismatch between weights and tokens')
  tokens_shape = tf.shape(tokens)
  batch_size = tokens_shape[0]
  num_tokens = tokens_shape[1]

  check.Same([tokens.dtype.base_dtype,
              weights.dtype.base_dtype],
             'dtype mismatch')

  # Flatten out the batch dimension so we can use one big matmul().
  tokens_bnxt = tf.reshape(tokens, [-1, num_activations])
  labels_bnxl = tf.matmul(tokens_bnxt, weights, transpose_b=True)

  # Restore the batch dimension in the output.
  labels_bxnxl = tf.reshape(labels_bnxl, [batch_size, num_tokens, num_labels])
  return labels_bxnxl


def LabelPotentialsFromTokenPairs(sources, targets, weights):
  r"""Computes label potentials from source and target tokens and weights.

  For each aligned pair of source and target token activations, computes a
  scalar potential for each label on the arc from the source to the target.
  Specifically,

    labels[b,t,l] = \sum_{i,j} sources[b,t,i] * weights[l,i,j] * targets[b,t,j]

  Args:
    sources: [B,N,S] tensor of batched source token activations.
    targets: [B,N,T] tensor of batched target token activations.
    weights: [L,S,T] tensor of weights.

    B,N may be dynamic, but L,S,T must be static.  The dtype of all arguments
    must be compatible.

  Returns:
    [B,N,L] tensor of label potentials as defined above, with the same dtype as
    the arguments.
  """
  check.Eq(sources.get_shape().ndims, 3, 'sources must be rank 3')
  check.Eq(targets.get_shape().ndims, 3, 'targets must be rank 3')
  check.Eq(weights.get_shape().ndims, 3, 'weights must be rank 3')

  num_labels = weights.get_shape().as_list()[0]
  num_source_activations = weights.get_shape().as_list()[1]
  num_target_activations = weights.get_shape().as_list()[2]
  check.NotNone(num_labels, 'unknown number of labels')
  check.NotNone(num_source_activations, 'unknown source activation dimension')
  check.NotNone(num_target_activations, 'unknown target activation dimension')
  check.Eq(sources.get_shape().as_list()[2], num_source_activations,
           'activation mismatch between weights and source tokens')
  check.Eq(targets.get_shape().as_list()[2], num_target_activations,
           'activation mismatch between weights and target tokens')

  check.Same([sources.dtype.base_dtype,
              targets.dtype.base_dtype,
              weights.dtype.base_dtype],
             'dtype mismatch')

  sources_shape = tf.shape(sources)
  targets_shape = tf.shape(targets)
  batch_size = sources_shape[0]
  num_tokens = sources_shape[1]
  with tf.control_dependencies([tf.assert_equal(batch_size, targets_shape[0]),
                                tf.assert_equal(num_tokens, targets_shape[1])]):
    # For each token, we must compute a vector-3tensor-vector product.  There is
    # no op for this, but we can use reshape() and matmul() to compute it.

    # Reshape |weights| and |targets| so we can use a single matmul().
    weights_lsxt = tf.reshape(weights, [num_labels * num_source_activations,
                                        num_target_activations])
    targets_bnxt = tf.reshape(targets, [-1, num_target_activations])
    weights_targets_bnxls = tf.matmul(targets_bnxt, weights_lsxt,
                                      transpose_b=True)

    # Restore all dimensions.
    weights_targets_bxnxlxs = tf.reshape(
        weights_targets_bnxls,
        [batch_size, num_tokens, num_labels, num_source_activations])

    # Incorporate the source activations.  In this case, we perform a batched
    # matmul() between the trailing [L,S] matrices of the current result and the
    # trailing [S] vectors of the tokens.
    sources_bxnx1xs = tf.expand_dims(sources, 2)
    labels_bxnxlx1 = tf.matmul(weights_targets_bxnxlxs, sources_bxnx1xs,
                               transpose_b=True)
    labels_bxnxl = tf.squeeze(labels_bxnxlx1, [3])
    return labels_bxnxl
