解释 cifar10.py 中常量的意思, and something else.

- [x] NUM_EPOCHS_PER_DECAY : 在 NUM_EPOCHS_PER_DECAY 个 epochs 之后，learning rate 开始 decay

- [x] decay_steps = int(num_batches_per_epoch * NUM_EPOCHS_PER_DECAY): decay_steps 步之后，开始 decay

- [x] global_step:  a TensorFlow variable that you increment at each training step 
- [x] 看看怎么 increment 了？应该是在 cifar10_train.py 中找找吧： apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
apply_gradients 会 increment global_step
- [ ] moving average 不知道是什么
- [x] 有点迷的语法：with tf.control_dependencies([loss_averages_op]):
                  with tf.control_dependencies([apply_gradient_op, variables_averages_op]): --- run 这个 context 里定义的 op 之前，必须执行作为参数的 ops 和 tensors.
