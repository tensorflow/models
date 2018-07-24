import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time
import functools


import google3 
import google3.third_party.tensorflow.python as tf
import google3.third_party.tensorflow.contrib.eager as tfe

from google3.third_party.tensorflow.python.keras.preprocessing import image
from google3.third_party.tensorflow.python.keras import models 
from google3.third_party.tensorflow.python.keras import losses
from google3.third_party.tensorflow.python.keras import layers
from google3.third_party.tensorflow.python.keras import backend as K


tf.enable_eager_execution()
print("Eager execution: {}".format(tf.executing_eagerly()))


img_shape = (512, 512, 3)
content_path = '/tmp/nst/dancing.png'
style_path = '/tmp/nst/picasso.png'


# # Let's begin by visualizing our input

# In[ ]:


def load_img(path_to_img):
  img = image.load_img(path_to_img, target_size=(img_shape[0], img_shape[1]))
  img = image.img_to_array(img)
  # We need to broadcast the image array such that it has a batch dimension 
  img = np.expand_dims(img, axis=0)
  return img


# In[ ]:


def imshow(img, title=None):
  # Remove the batch dimension
  out = np.squeeze(img, axis=0)
  # Normalize for display 
  out = out.astype('uint8')
  plt.imshow(out)
  if title is not None:
    plt.title(title)
  plt.imshow(out)


# These are input content and style images. We hope to "create" an image with
# the content of our content image, but with the style of the style image.

# In[7]:


plt.figure(figsize=(10,10))

content = load_img(content_path).astype('uint8')
style = load_img(style_path)

plt.subplot(1, 2, 1)
imshow(content, 'Content Image')

plt.subplot(1, 2, 2)
imshow(style, 'Style Image')
plt.show()




def load_and_process_img(path_to_img):
  img = load_img(path_to_img)
  img = tf.keras.applications.vgg19.preprocess_input(img)
  return img




def deprocess_img(processed_img):
  x = processed_img.copy()
  if len(x.shape) == 4:
    x = x.reshape(img_shape)
  assert len(x.shape) == 3, ("Input to deprocess image must be an image of "
                             "dimension [batch, height, width, channel] or [height_width_channel]")
  if len(x.shape) != 3:
    raise ValueError("Invalid input to deprocessing image")
  
  # perform the inverse of the preprocessiing step
  x[:, :, 0] += 103.939
  x[:, :, 1] += 116.779
  x[:, :, 2] += 123.68
  x = x[:, :, ::-1]

  x = np.clip(x, 0, 255).astype('uint8')
  return x



# Content layer where will pull our feature maps
content_layers = ['block5_conv2'] 

# Style layer we are interested in
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1', 
                'block4_conv1', 
                'block5_conv1'
               ]

num_content_layers = len(content_layers)
num_style_layers = len(style_layers)




def get_model():
  """ Creates our model with access to intermediate layers. 
  
  This function will load the VGG19 model and access the intermediate layers. 
  These layers will then be used to create a new model that will take input image
  and return the outputs from these intermediate layers from the VGG model. 
  
  Returns:
    returns a keras model that takes image inputs and outputs the style and 
      content intermediate layers. 
  """
  # Load our model. We load pretrained VGG, trained on imagenet data
  vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False
  # Get output layers corresponding to style and content layers 
  style_outputs = [vgg.get_layer(name).output for name in style_layers]
  content_outputs = [vgg.get_layer(name).output for name in content_layers]
  model_outputs = style_outputs + content_outputs
  # Build model 
  return models.Model(vgg.input, model_outputs)





def get_content_loss(base_content, target):
  return tf.reduce_mean(tf.square(base_content - target))




def gram_matrix(input_tensor):
  # We make the image channels first 
  channels = int(input_tensor.shape[-1])
  a = tf.reshape(input_tensor, [-1, channels])
  n = tf.shape(a)[0]
  gram = tf.matmul(a, a, transpose_a=True)
  return gram / tf.cast(n, tf.float32)

def get_style_loss(base_style, gram_target):
  """Expects two images of dimension h, w, c"""
  # height, width, num filters of each layer
  # We scale the loss at a given layer by the size of the feature map and the number of filters
  height, width, channels = base_style.get_shape().as_list()
  gram_style = gram_matrix(base_style)
  
  return tf.reduce_mean(tf.square(gram_style - gram_target))# / (4. * (channels ** 2) * (width * height) ** 2)





def get_feature_representations(model, content_path, style_path):
  """Helper function to compute our content and style feature representations.

  This function will simply load and preprocess both the content and style 
  images from their path. Then it will feed them through the network to obtain
  the outputs of the intermediate layers. 
  
  Arguments:
    model: The model that we are using.
    content_path: The path to the content image.
    style_path: The path to the style image
    
  Returns:
    returns the style features and the content features. 
  """
  # Load our images in 
  content_image = load_and_process_img(content_path)
  style_image = load_and_process_img(style_path)
  
  # batch compute content and style features
  stack_images = np.concatenate([style_image, content_image], axis=0)
  model_outputs = model(stack_images)
  
  # Get the style and content feature representations from our model  
  style_features = [style_layer[0] for style_layer in model_outputs[:num_style_layers]]
  content_features = [content_layer[1] for content_layer in model_outputs[num_style_layers:]]
  return style_features, content_features




def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
  """This function will compute the loss total loss.
  
  Arguments:
    model: The model that will give us access to the intermediate layers
    loss_weights: The weights of each contribution of each loss function. 
      (style weight, content weight, and total variation weight)
    init_image: Our initial base image. This image is what we are updating with 
      our optimization process. We apply the gradients wrt the loss we are 
      calculating to this image.
    gram_style_features: Precomputed gram matrices corresponding to the 
      defined style layers of interest.
    content_features: Precomputed outputs from defined content layers of 
      interest.
      
  Returns:
    returns the total loss, style loss, content loss, and total variational loss
  """
  style_weight, content_weight = loss_weights
  
  # Feed our init image through our model. This will give us the content and 
  # style representations at our desired layers. Since we're using eager
  # our model is callable just like any other function!
  model_outputs = model(init_image)
  
  style_output_features = model_outputs[:num_style_layers]
  content_output_features = model_outputs[num_style_layers:]
  
  style_score = 0
  content_score = 0

  # Accumulate style losses from all layers
  # Here, we equally weight each contribution of each loss layer
  weight_per_style_layer = 1.0 / float(num_style_layers)
  for target_style, comb_style in zip(gram_style_features, style_output_features):
    style_score += weight_per_style_layer * get_style_loss(comb_style[0], target_style)
    
  # Accumulate content losses from all layers 
  weight_per_content_layer = 1.0 / float(num_content_layers)
  for target_content, comb_content in zip(content_features, content_output_features):
    content_score += weight_per_content_layer* get_content_loss(comb_content[0], target_content)
  
  style_score *= style_weight
  content_score *= content_weight

  # Get total loss
  loss = style_score + content_score 
  return loss, style_score, content_score



def compute_grads(cfg):
  with tf.GradientTape() as tape: 
    all_loss = compute_loss(**cfg)
  # Compute gradients wrt input image
  total_loss = all_loss[0]
  return tape.gradient(total_loss, cfg['init_image']), all_loss


# ## Apply and run the style transfer process




def run_style_transfer(content_path, 
                       style_path,
                       num_iterations=1000,
                       content_weight=1e3, 
                       style_weight=1e-2): 
  display_num = 100
  # We don't need to (or want to) train any layers of our model, so we set their
  # trainable to false. 
  model = get_model() 
  for layer in model.layers:
    layer.trainable = False
  
  # Get the style and content feature representations (from our specified intermediate layers) 
  style_features, content_features = get_feature_representations(model, content_path, style_path)
  gram_style_features = [gram_matrix(style_feature) for style_feature in style_features]
  
  # Set initial image
  init_image = load_and_process_img(content_path)
  init_image = tfe.Variable(init_image, dtype=tf.float32)
  # Create our optimizer
  opt = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)

  # For displaying intermediate images 
  iter_count = 1
  
  # Store our best result
  best_loss, best_img = float('inf'), None
  
  # Create a nice config 
  loss_weights = (style_weight, content_weight)
  cfg = {
      'model': model,
      'loss_weights': loss_weights,
      'init_image': init_image,
      'gram_style_features': gram_style_features,
      'content_features': content_features
  }
    
  # For displaying
  plt.figure(figsize=(14, 7))
  num_rows = (num_iterations / display_num) // 5
  start_time = time.time()
  global_start = time.time()
  
  norm_means = np.array([103.939, 116.779, 123.68])
  min_vals = -norm_means
  max_vals = 255 - norm_means   
  for i in range(num_iterations):
    grads, all_loss = compute_grads(cfg)
    loss, style_score, content_score = all_loss
    # grads, _ = tf.clip_by_global_norm(grads, 5.0)
    opt.apply_gradients([(grads, init_image)])
    clipped = tf.clip_by_value(init_image, min_vals, max_vals)
    init_image.assign(clipped)
    end_time = time.time() 
    
    if loss < best_loss:
      # Update best loss and best image from total loss. 
      best_loss = loss
      best_img = init_image.numpy()

    if i % display_num == 0:
      print('Iteration: {}'.format(i))        
      print('Total loss: {:.4e}, ' 
            'style loss: {:.4e}, '
            'content loss: {:.4e}, '
            'time: {:.4f}s'.format(loss, style_score, content_score, time.time() - start_time))
      start_time = time.time()
      
      # Display intermediate images
      if iter_count > num_rows * 5: continue 
      plt.subplot(num_rows, 5, iter_count)
      # Use the .numpy() method to get the concrete numpy array
      plot_img = init_image.numpy()
      plot_img = deprocess_img(plot_img)
      
      plt.imshow(plot_img)
      plt.title('Iteration {}'.format(i + 1))
      iter_count += 1
  print('Total time: {:.4f}s'.format(time.time() - global_start))
      
  return best_img, best_loss 


# In[18]:


best, best_loss = run_style_transfer(content_path, 
                                     style_path)


# # Visualize outputs
# We "deprocess" the output image in order to remove the processing that was applied to it. 

# In[ ]:


def show_results(best_img, content_path, style_path, show_large_final=True):
  plt.figure(figsize=(15, 15))
  x = deprocess_img(best_img)
  content = load_img(content_path) 
  style = load_img(style_path)

  plt.subplot(1, 3, 1)
  imshow(content, 'Content Image')

  plt.subplot(1, 3, 2)
  imshow(style, 'Style Image')

  plt.subplot(1, 3, 3)
  plt.imshow(x)
  plt.title('Output Image')
  plt.show()

  if show_large_final: 
    plt.figure(figsize=(10, 10))

    plt.imshow(x)
    plt.title('Output Image')
    plt.show()


# In[20]:


show_results(best, content_path, style_path)


# ## Let's see some other applications!

# ## Starry night + Tubingen

# In[21]:


best_starry_night, best_loss = run_style_transfer('/tmp/nst/tubingen.png',
                                                  '/tmp/nst/starry_night.png')


# In[22]:


show_results(best_starry_night, '/tmp/nst/tubingen.png', '/tmp/nst/starry_night.png')


# ## Picasso + Tubingen

# In[23]:


best_picasso, best_loss = run_style_transfer('/tmp/nst/tubingen.png', 
                                             '/tmp/nst/picasso.png')


# In[24]:


show_results(best_picasso, '/tmp/nst/tubingen.png', '/tmp/nst/picasso.png')


# ## Dancer + Starry Night

# In[25]:


best_starry_dance, best_loss = run_style_transfer('/tmp/nst/dancing.png', 
                                                  '/tmp/nst/Pillars_of_creation_2014_HST_WFC3-UVIS_full-res_denoised.jpg')


# In[26]:


show_results(best_starry_dance, '/tmp/nst/dancing.png', '/tmp/nst/Pillars_of_creation_2014_HST_WFC3-UVIS_full-res_denoised.jpg')


# In[27]:


best_poc_tubingen, best_loss = run_style_transfer('/tmp/nst/tubingen.png', 
                                                  '/tmp/nst/Pillars_of_creation_2014_HST_WFC3-UVIS_full-res_denoised.jpg')


# In[28]:


show_results(best_poc_tubingen, 
             '/tmp/nst/tubingen.png',
             '/tmp/nst/Pillars_of_creation_2014_HST_WFC3-UVIS_full-res_denoised.jpg')

