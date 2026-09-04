"""
Efficient Data Loading Pipeline for FCOS using tf.data.Dataset.
"""

import sys
import re
from operator import itemgetter
import json
import tensorflow as tf
from official.projects.fcos.utils.concatenate import concat
import os
import numpy as np

# COCO 2017 Constants
# Note: FCOS requires specific target encoding for each FPN level.
# This implementation provides the data loading and preprocessing pipeline.
# The `encode_targets` function is where the matching logic (ground truth -> FPN levels) resides.

IMG_HEIGHT = 800
IMG_WIDTH = 1024
MAX_BOXES_PER_IMAGE = 100

def load_coco_annotations(json_path, image_dir, max_images=None):
    """
    Parses COCO JSON annotations and returns lists of image paths and formatted annotations.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    images = {img['id']: img for img in data['images']}
    if max_images:
        # Slice dictionary to limit size
        import itertools
        images = dict(itertools.islice(images.items(), max_images))

    annotations = {}
    
    for ann in data['annotations']:
        img_id = ann['image_id']
        if img_id not in annotations:
            annotations[img_id] = []
        
        # COCO bbox format: [x, y, width, height]
        # We might want to convert to [y_min, x_min, y_max, x_max] or similar depending on model needs
        bbox = ann['bbox']
        category_id = ann['category_id']
        annotations[img_id].append(bbox + [category_id])

    image_paths = []
    bboxes = []
    
    for img_id, img_data in images.items():
        if img_id in annotations:
             # Construct full path
            path = os.path.join(image_dir, img_data['file_name'])
            image_paths.append(path)
            
            # Pad or truncate boxes to fixed size for tensor compatibility
            boxes_raw = annotations[img_id]
            if len(boxes_raw) > MAX_BOXES_PER_IMAGE:
                boxes_raw = boxes_raw[:MAX_BOXES_PER_IMAGE]
            else:
                # Pad with zeros
                boxes_raw += [[0.0, 0.0, 0.0, 0.0, 0]] * (MAX_BOXES_PER_IMAGE - len(boxes_raw))
            
            bboxes.append(boxes_raw)
    
    # Clear large dictionaries to free RAM immediately
    del data
    del images
    del annotations
    import gc
    gc.collect()
            
    return image_paths, bboxes

def preprocess_image(path, boxes):
    """
    Reads image, resizes it, and normalizes it.
    Also adjusts bounding boxes to match the new image size.
    """
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    
    # Original dimensions
    original_shape = tf.shape(image)
    h_original = tf.cast(original_shape[0], tf.float32)
    w_original = tf.cast(original_shape[1], tf.float32)
    
    # Resize
    image = tf.image.resize(image, [IMG_HEIGHT, IMG_WIDTH])
    image = image / 255.0  # Normalize to [0, 1]
    
    # Adjust boxes: [x, y, w, h] -> need to scale x by (NEW_W / OLD_W), etc.
    # boxes shape: (MAX_BOXES, 5) where 5 is [x, y, w, h, class]
    boxes_float = tf.cast(boxes, tf.float32)
    
    scale_x = IMG_WIDTH / w_original
    scale_y = IMG_HEIGHT / h_original
    
    # Split boxes to apply scaling
    x = boxes_float[:, 0] * scale_x
    y = boxes_float[:, 1] * scale_y
    w = boxes_float[:, 2] * scale_x
    h = boxes_float[:, 3] * scale_y
    classes = boxes_float[:, 4]
    
    scaled_boxes = tf.stack([x, y, w, h, classes], axis=-1)
    
    return image, scaled_boxes

def produce_grid(h, w, stride):
    """
    Generates a grid of (x, y) coordinates for a given feature map size.
    Corresponds to the input image coordinates of the feature map pixels.
    """
    shift_x = tf.range(w) * stride + stride // 2
    shift_y = tf.range(h) * stride + stride // 2
    
    # Create grid mesh
    shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
    
    # Flatten to (h*w, 2)
    grid = tf.stack([tf.reshape(shift_x, [-1]), tf.reshape(shift_y, [-1])], axis=-1)
    return tf.cast(grid, dtype=tf.float32)

def encode_targets(image, boxes, strides=[8, 16, 32, 64, 128], limit_ranges=[[0, 64], [64, 128], [128, 256], [256, 512], [512, 99999]]):
    """
    Core FCOS Logic: Maps ground truth boxes to feature map levels (P3-P7).
    
    Args:
        image: A tensor of shape (H, W, 3)
        boxes: A tensor of shape (N, 5) -> [x_min, y_min, w, h, class_id]
               (Note: input boxes are padded to fixed size N)
    
    Returns:
        image: Original image
        targets: Dict containing 'classifier', 'box', 'centerness' dense tensors.
    """
    
    # 1. Separate valid boxes from padding
    # Padding was done with [0,0,0,0,0]. Valid boxes have w > 0 and h > 0.
    # We can detect valid boxes by checking class_id != 0 or width > 0.
    # Assuming valid class_ids are >= 1.
    
    # Convert boxes to [x_min, y_min, x_max, y_max, class_id] for convenience
    # Input is [x, y, w, h, class]
    img_h, img_w = IMG_HEIGHT, IMG_WIDTH
    
    x_min = boxes[:, 0]
    y_min = boxes[:, 1]
    w = boxes[:, 2]
    h = boxes[:, 3]
    class_ids = boxes[:, 4]
    
    x_max = x_min + w
    y_max = y_min + h
    
    # Calculate centers and areas of GT boxes
    gt_centers_x = x_min + w / 2.0
    gt_centers_y = y_min + h / 2.0
    gt_areas = w * h
    
    all_class_targets = []
    all_box_targets = []
    all_centerness_targets = []
    
    # Iterate over each FPN level
    for stride, limit_range in zip(strides, limit_ranges):
        # Calculate feature map size for this level
        fm_h = tf.math.ceil(img_h / stride)
        fm_w = tf.math.ceil(img_w / stride)
        fm_h = tf.cast(fm_h, tf.int32)
        fm_w = tf.cast(fm_w, tf.int32)
        
        # Generate grid points for this level
        # Shape: (points, 2)
        locations = produce_grid(fm_h, fm_w, stride)
        num_points = tf.shape(locations)[0]
        
        # --- CALCULATE REGRESSION TARGETS FOR ALL PAIRS ---
        # locations: (points, 1, 2)
        # gt_box_coords: (1, num_boxes, 2)
        
        # Left: x - x_min
        # Top: y - y_min
        # Right: x_max - x
        # Bottom: y_max - y
        
        # Expansion for broadcasting: (Points, Boxes)
        dx_min = locations[:, 0][:, None] - x_min[None, :]
        dy_min = locations[:, 1][:, None] - y_min[None, :]
        dx_max = x_max[None, :] - locations[:, 0][:, None]
        dy_max = y_max[None, :] - locations[:, 1][:, None]
        
        # Stack into (Points, Boxes, 4): [l, t, r, b]
        ltrb = tf.stack([dx_min, dy_min, dx_max, dy_max], axis=-1)
        
        # --- CHECK CONDITIONS ---
        
        # 1. Inside Box: min(l,t,r,b) > 0
        is_in_box = tf.reduce_min(ltrb, axis=-1) > 0
        
        # 2. Scale Constraint: max(l,t,r,b) must be in [min_limit, max_limit]
        max_reg = tf.reduce_max(ltrb, axis=-1)
        is_in_level = tf.logical_and(max_reg > limit_range[0], max_reg <= limit_range[1])
        
        # Combined valid mask (Points, Boxes)
        # Also filter out padded dummy boxes (class_id == 0)
        is_valid_box = class_ids > 0
        valid_match = tf.logical_and(tf.logical_and(is_in_box, is_in_level), is_valid_box[None, :])
        
        # --- AMBIGUITY RESOLUTION: SELECT MIN AREA BOX ---
        
        # Set area of invalid matches to infinity so they aren't picked
        areas_broadcast = tf.broadcast_to(gt_areas[None, :], tf.shape(valid_match))
        inf = 1e8
        masked_areas = tf.where(valid_match, areas_broadcast, inf)
        
        # Find index of box with min area for each point
        min_area_vals = tf.reduce_min(masked_areas, axis=1)
        min_area_idx = tf.argmin(masked_areas, axis=1) # (Points,)
        
        # Points that matched at least one valid box
        targets_mask = min_area_vals < inf
        
        # --- GATHER FINAL TARGETS ---
        
        # Gather the chosen box attributes for each point
        # class_ids: (Boxes,) -> gather -> (Points,)
        assigned_classes = tf.gather(class_ids, min_area_idx)
        
        # ltrb: (Points, Boxes, 4) -> gather -> (Points, 4)
        # Note: tf.gather_nd is needed or careful indexing. 
        # Easier: Gather from (Boxes, 4) using indices
        # We need to re-compute or store ltrb for the winning boxes?
        # Re-calc matches better for simplicity or gather from boxes?
        # Let's gather the winning box coords and re-calc ltrb for the single winning box is safer/cleaner.
        
        min_x = tf.gather(x_min, min_area_idx)
        min_y = tf.gather(y_min, min_area_idx)
        max_x = tf.gather(x_max, min_area_idx)
        max_y = tf.gather(y_max, min_area_idx)
        
        # Recalculate ltrb for the winning box
        l = locations[:, 0] - min_x
        t = locations[:, 1] - min_y
        r = max_x - locations[:, 0]
        b = max_y - locations[:, 1]
        
        reg_targets = tf.stack([l, t, r, b], axis=-1)
        
        # Apply mask: Background points get 0
        # Classes
        final_cls_ids = tf.where(targets_mask, assigned_classes, tf.zeros_like(assigned_classes))
        
        # Regression
        # Note: FCOS often normalizes regression targets by stride (optional but recommended)
        # We will normalize by stride here to keep gradients stable.
        reg_targets = reg_targets / stride
        final_reg_targets = tf.where(targets_mask[:, None], reg_targets, tf.zeros_like(reg_targets))
        
        # Centerness
        # c = sqrt( (min(l,r)/max(l,r)) * (min(t,b)/max(t,b)) )
        l_r_min = tf.math.minimum(l, r)
        l_r_max = tf.math.maximum(l, r)
        t_b_min = tf.math.minimum(t, b)
        t_b_max = tf.math.maximum(t, b)
        
        centerness = tf.sqrt((l_r_min / l_r_max) * (t_b_min / t_b_max))
        final_centerness = tf.where(targets_mask, centerness, tf.zeros_like(centerness))
        # Ensure dimensionality (Points, 1)
        final_centerness = final_centerness[:, None]
        
        # --- FORMATTING ---
        # Classification needs to be one-hot? 
        # Model output is (Batch, Points, 80). Loss 'CategoricalFocalCrossentropy' expects one-hot.
        # class_ids are 1-80. We need one-hot 80 classes.
        # But wait, class_id 0 is background.
        # The formulation usually uses 'sigmoid focal loss' on (num_classes) outputs.
        # If we use CategoricalFocalCrossentropy, we need 80 classes.
        # Let's assume indices 0-79. So we map class_id -> class_id - 1.
        # If class_id is 0 (BG), we should produce a vector of all zeros corresponding to "no object".
        # But standard Focal Loss implementation usually expects background as a separate handling or implicit.
        # Let's produce one-hot for classes 1-80.
        
        # Valid classes: 1..80.
        # Target index: 0..79.
        # IF bg, we want all zeros? Or a specific background class? 
        # Typical RetinaNet/FCOS approach: C binary classifiers.
        # If we use Categorical, we imply Softmax? FCOS uses Sigmoid per class usually.
        # The user's code uses `CategoricalFocalCrossentropy` which implies Softmax normalization over classes or "from_logits=True".
        # Let's produce One Hot.
        
        # Shift class ids: 1->0 ... 80->79
        cls_indices = tf.cast(final_cls_ids - 1, tf.int32)
        one_hot = tf.one_hot(cls_indices, depth=80)
        
        # Zero out background rows completely
        # (tf.one_hot handles -1 by zeroing output? No, usually not. We must mask.)
        one_hot = tf.where(targets_mask[:, None], one_hot, tf.zeros_like(one_hot))
        
        all_class_targets.append(one_hot)
        all_box_targets.append(final_reg_targets)
        all_centerness_targets.append(final_centerness)

    # Concatenate all levels
    # Shape: (Total_Points, 80)
    cls_concat = tf.concat(all_class_targets, axis=0)
    # Shape: (Total_Points, 4)
    box_concat = tf.concat(all_box_targets, axis=0)
    # Shape: (Total_Points, 1)
    ctr_concat = tf.concat(all_centerness_targets, axis=0)
    
    return image, {'classifier': cls_concat, 'box': box_concat, 'centerness': ctr_concat}

def get_training_dataset(train_imgs_path, annotations_path, batch_size, max_images=None):
    """
    Creates the tf.data.Dataset pipeline.
    """
    image_paths, all_boxes = load_coco_annotations(annotations_path, train_imgs_path, max_images=max_images)
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, all_boxes))
    
    # Shuffle
    dataset = dataset.shuffle(buffer_size=1000)
    
    # Map preprocessing (Parallelized)
    dataset = dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Map targets
    dataset = dataset.map(encode_targets, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Batch
    dataset = dataset.batch(batch_size, drop_remainder=True)
    
    # Prefetch
    # Limit prefetch buffer to conserve RAM (AUTOTUNE can grow unbounded)
    dataset = dataset.prefetch(4)
    
    return dataset
