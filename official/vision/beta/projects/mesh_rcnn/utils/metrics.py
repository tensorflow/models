import tensorflow as tf

def compare_meshes(
    pred_meshes, gt_meshes, num_samples=10000, scale="gt-10", thresholds=None, reduce=True, eps=1e-8
):

    if thresholds is None:
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5]

    pred_meshes, gt_meshes = _scale_meshes(pred_meshes, gt_meshes, scale)

    if isinstance(num_samples, tuple):
        num_samples_pred, num_samples_gt = num_samples
    else:
        num_samples_pred = num_samples_gt = num_samples

    pred_points, pred_normals = _sample_meshes(pred_meshes, num_samples_pred)
    gt_points, gt_normals = _sample_meshes(gt_meshes, num_samples_gt)
    if pred_points is None:
        logger.info("WARNING: Sampling predictions failed during eval")
        return None
    elif gt_points is None:
        logger.info("WARNING: Sampling GT failed during eval")
        return None

    if len(gt_meshes) == 1:
        # (1, S, 3) to (N, S, 3)
        gt_points = tf.broadcast_to(gt_points, [len(pred_meshes), num_samples, 3])
        gt_normals = tf.broadcast_to(gt_normals, [len(pred_meshes), num_samples, 3])
        #gt_points = gt_points.expand(len(pred_meshes), -1, -1)
        #gt_normals = gt_normals.expand(len(pred_meshes), -1, -1)

    if tf.is_tensor(pred_points) and tf.is_tensor(gt_points):
        # We can compute all metrics at once in this case
        metrics = _compute_sampling_metrics(
            pred_points, pred_normals, gt_points, gt_normals, thresholds, eps
        )
    else:
        # Slow path when taking vert samples from non-equisized meshes; we need
        # to iterate over the batch
        metrics = defaultdict(list)
        for cur_points_pred, cur_points_gt in zip(pred_points, gt_points):
            cur_metrics = _compute_sampling_metrics(
                cur_points_pred[None], None, cur_points_gt[None], None, thresholds, eps
            )
            for k, v in cur_metrics.items():
                metrics[k].append(v.item())
        metrics = {k: tf.tensor(vs) for k, vs in metrics.items()}

    if reduce:
        # Average each metric over the batch
        metrics = {k: v.mean().item() for k, v in metrics.items()}

    return metrics

def _scale_meshes(pred_meshes, gt_meshes, scale):
    if isinstance(scale, float):
        # Assume scale is a single scalar to use for both preds and GT
        pred_scale = gt_scale = scale
    elif isinstance(scale, tuple):
        # Rescale preds and GT with different scalars
        pred_scale, gt_scale = scale
    elif scale.startswith("gt-"):
        # Rescale both preds and GT so that the largest edge length of each GT
        # mesh is target
        target = float(scale[3:])
        bbox = gt_meshes.get_bounding_boxes()  # (N, 3, 2)
        long_edge = (bbox[:, :, 1] - bbox[:, :, 0]).max(dim=1)[0]  # (N,)
        scale = target / long_edge
        if scale.numel() == 1:
            scale = tf.broadcast_to(scale, [len(pred_meshes)])
            #scale = scale.expand(len(pred_meshes))
        pred_scale, gt_scale = scale, scale
    else:
        raise ValueError("Invalid scale: %r" % scale)
    pred_meshes = pred_meshes.scale_verts(pred_scale)
    gt_meshes = gt_meshes.scale_verts(gt_scale)
    return pred_meshes, gt_meshes


def get_bounding_boxes(mesh):
        """
        Compute an axis-aligned bounding box for each mesh in this Meshes object.

        Returns:
            bboxes: Tensor of shape (N, 3, 2) where bbox[i, j] gives the
            min and max values of mesh i along the jth coordinate axis.
        """
        all_mins, all_maxes = [], []
        for verts in mesh.verts():
            cur_mins = verts.min(dim=0)[0]  # (3,)
            cur_maxes = verts.max(dim=0)[0]  # (3,)
            all_mins.append(cur_mins)
            all_maxes.append(cur_maxes)
        all_mins = tf.stack(all_mins, axis=0)  # (N, 3)
        all_maxes = tf.stack(all_maxes, axis=0)  # (N, 3)
        bboxes = tf.stack([all_mins, all_maxes], axis=2)
        return bboxes

def _compute_sampling_metrics(pred_points, pred_normals, gt_points, gt_normals, thresholds, eps):
   
    metrics = {}
    lengths_pred = tf.fill(
        (pred_points.shape[0],), pred_points.shape[1], dtype=torch.int64, device=pred_points.device
    )
    lengths_gt = tf.fill(
        (gt_points.shape[0],), gt_points.shape[1], dtype=torch.int64, device=gt_points.device
    )

    # For each predicted point, find its neareast-neighbor GT point
    knn_pred = knn_points(pred_points, gt_points, lengths1=lengths_pred, lengths2=lengths_gt, K=1)
    # Compute L1 and L2 distances between each pred point and its nearest GT
    pred_to_gt_dists2 = knn_pred.dists[..., 0]  # (N, S)
    pred_to_gt_dists = pred_to_gt_dists2.sqrt()  # (N, S)
    if gt_normals is not None:
        pred_normals_near = knn_gather(gt_normals, knn_pred.idx, lengths_gt)[..., 0, :]  # (N, S, 3)
    else:
        pred_normals_near = None

    # For each GT point, find its nearest-neighbor predicted point
    knn_gt = knn_points(gt_points, pred_points, lengths1=lengths_gt, lengths2=lengths_pred, K=1)
    # Compute L1 and L2 dists between each GT point and its nearest pred point
    gt_to_pred_dists2 = knn_gt.dists[..., 0]  # (N, S)
    gt_to_pred_dists = gt_to_pred_dists2.sqrt()  # (N, S)

    if pred_normals is not None:
        gt_normals_near = knn_gather(pred_normals, knn_gt.idx, lengths_pred)[..., 0, :]  # (N, S, 3)
    else:
        gt_normals_near = None

    # Compute L2 chamfer distances
    chamfer_l2 = pred_to_gt_dists2.mean(dim=1) + gt_to_pred_dists2.mean(dim=1)
    metrics["Chamfer-L2"] = chamfer_l2

    # Compute normal consistency and absolute normal consistance only if
    # we actually got normals for both meshes
    if pred_normals is not None and gt_normals is not None:
        pred_to_gt_cos = F.cosine_similarity(pred_normals, pred_normals_near, dim=2)
        gt_to_pred_cos = F.cosine_similarity(gt_normals, gt_normals_near, dim=2)

        pred_to_gt_cos_sim = pred_to_gt_cos.mean(dim=1)
        pred_to_gt_abs_cos_sim = pred_to_gt_cos.abs().mean(dim=1)
        gt_to_pred_cos_sim = gt_to_pred_cos.mean(dim=1)
        gt_to_pred_abs_cos_sim = gt_to_pred_cos.abs().mean(dim=1)
        normal_dist = 0.5 * (pred_to_gt_cos_sim + gt_to_pred_cos_sim)
        abs_normal_dist = 0.5 * (pred_to_gt_abs_cos_sim + gt_to_pred_abs_cos_sim)
        metrics["NormalConsistency"] = normal_dist
        metrics["AbsNormalConsistency"] = abs_normal_dist

    # Compute precision, recall, and F1 based on L2 distances
    for t in thresholds:
        precision = 100.0 * (pred_to_gt_dists < t).float().mean(dim=1)
        recall = 100.0 * (gt_to_pred_dists < t).float().mean(dim=1)
        f1 = (2.0 * precision * recall) / (precision + recall + eps)
        metrics["Precision@%f" % t] = precision
        metrics["Recall@%f" % t] = recall
        metrics["F1@%f" % t] = f1

    # Move all metrics to CPU
    metrics = {k: v.cpu() for k, v in metrics.items()}
    return metrics