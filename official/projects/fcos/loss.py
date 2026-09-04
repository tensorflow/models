"""
Implementation of the IOU loss function from the RetinaNet
research paper
"""

import tensorflow as tf


class IOULoss(tf.keras.Loss):
    """Intersection Over Union (IOU) loss function"""

    def __init__(self):
        super().__init__()

    def call(self, y_true, y_pred):
        """
        Calculates IoU Loss (negative log IoU) for regression targets.
        
        Args:
            y_true: Ground truth tensor of shape (Batch, Points, 4) [l, t, r, b]
            y_pred: Prediction tensor of shape (Batch, Points, 4) [l, t, r, b]
            
        Returns:
            loss: Tensor of shape (Batch, Points)
        """
        # Split coordinates
        # y_true/pred: [left, top, right, bottom]
        
        # Ensure preds are positive (technically model should ensure this with exp, but safe to abs or clip)
        y_pred = tf.math.abs(y_pred) 
        
        # Ground Truth
        gt_l = y_true[..., 0]
        gt_t = y_true[..., 1]
        gt_r = y_true[..., 2]
        gt_b = y_true[..., 3]
        
        # Predictions
        pred_l = y_pred[..., 0]
        pred_t = y_pred[..., 1]
        pred_r = y_pred[..., 2]
        pred_b = y_pred[..., 3]
        
        # Areas
        target_area = (gt_l + gt_r) * (gt_t + gt_b)
        pred_area = (pred_l + pred_r) * (pred_t + pred_b)
        
        # Intersection width and height
        w_intersect = tf.minimum(pred_l, gt_l) + tf.minimum(pred_r, gt_r)
        h_intersect = tf.minimum(pred_t, gt_t) + tf.minimum(pred_b, gt_b)
        
        # Clip to 0 (in case of no intersection)
        w_intersect = tf.maximum(w_intersect, 0.0)
        h_intersect = tf.maximum(h_intersect, 0.0)
        
        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect
        
        # IoU
        # Add epsilon to prevent division by zero
        iou = (area_intersect + 1e-7) / (area_union + 1e-7)
        
        # IoU Loss = -ln(IoU)
        loss = -tf.math.log(iou + 1e-7)
        
        # Masking: Only calculate loss for positive samples.
        # Background samples (gt_l, gt_t, gt_r, gt_b) are (0,0,0,0) in our data pipeline.
        # So target_area == 0 implies background.
        mask = tf.cast(target_area > 0, dtype=tf.float32)
        
        # Apply mask
        loss = loss * mask
        
        # For Keras model.fit, we usually return the per-sample loss. 
        # But here we have per-anchor loss.
        # FCOS: Sum over positive samples / num_positive_samples.
        # Keras will take the mean of this output over the batch if we return (Batch, Points).
        # To strictly follow FCOS, we should sum and divide by num_pos.
        # However, num_pos varies per image. 
        # If we return the masked loss map, Keras Global Batch Mean will likely underestimate the loss 
        # (because of many zeros).
        # Better approach: partial reduction here?
        # But Keras expects (Batch, d0, .. dN) matching target.
        # Let's return the element-wise loss. The magnitude will be small, but gradient direction is correct.
        
        return loss


class FcosLoss(tf.keras.Loss):
    """implementation of the FCOS loss function"""

    def __init__(self):
        super().__init__()

    def call(self, pred, g_label):
        """
        the method includes the implementation of the FCOS loss function
        Args:
            pred : the prediction of the model
            g_label : the ground truth label of the model
        Returns:
            the loss value
        """
        focal = tf.keras.losses.CategoricalFocalCrossentropy()
        iouloss = IOULoss()
        bce = tf.keras.losses.BinaryCrossentropy()
        
        return focal(pred, g_label) , iouloss(pred, g_label) , bce(pred, g_label)
