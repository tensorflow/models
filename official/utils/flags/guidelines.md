# Using flags in official models

1. **All common flags must be incorporated in the models.**

   Common flags (i.e. batch_size, model_dir, etc.) are provided by various flag definition functions,
   and channeled through `official.utils.flags.core`. For instance to define common supervised
   learning parameters one could use the following code:

   ```$xslt
   from absl import app as absl_app
   from absl import flags

   from official.utils.flags import core as flags_core


   def define_flags():
     flags_core.define_base()
     flags.adopt_key_flags(flags_core)


   def main(_):
     flags_obj = flags.FLAGS
     print(flags_obj)


   if __name__ == "__main__"
     absl_app.run(main)
   ```
2. **Validate flag values.**

   See the [Validators](#validators) section for implementation details.

   Validators in the official model repo should not access the file system, such as verifying
   that files exist, due to the strict ordering requirements.

3. **Flag values should not be mutated.**

   Instead of mutating flag values, use getter functions to return the desired values. An example
   getter function is `get_loss_scale` function below:

   ```
   # Map string to (TensorFlow dtype, default loss scale)
   DTYPE_MAP = {
       "fp16": (tf.float16, 128),
       "fp32": (tf.float32, 1),
   }


   def get_loss_scale(flags_obj):
     if flags_obj.loss_scale == "dynamic":
       return flags_obj.loss_scale
     if flags_obj.loss_scale is not None:
       return flags_obj.loss_scale
     return DTYPE_MAP[flags_obj.dtype][1]


   def main(_):
     flags_obj = flags.FLAGS()

     # Do not mutate flags_obj
     # if flags_obj.loss_scale is None:
     #   flags_obj.loss_scale = DTYPE_MAP[flags_obj.dtype][1] # Don't do this

     print(get_loss_scale(flags_obj))
     ...
   ```