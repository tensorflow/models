import tensorflow as tf

# Import MNIST data
"""
One Hot Encoding
You may notice that when we are reading the datasets we set a parameter called one_hot to true. One hot encoding is generally used for classification. A one hot encoded set of outputs has a bit that can be either 0 or 1 for each output. In this case we have the numbers 0-9 we therefore have 10 output values where 4 is reprisented as [0,0,0,0,1,0,0,0,0,0] This is done to make it easier for the network to train.
"""
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

"""
Hyperparameters
We are still using the same parameters as we did before however we do have some new things to play with. We can now decide how many layers we want and how many nodes are in each layer!
There is no real rule as to how many layers / nodes you should have but generally you should start off small and work your way up. This will prevent overfitting. Here we will use two layers, each with 256 nodes. In reality this is overkill but the testing accuracy ended up high so i'm not complaining. It should also be noted that similar accuracys can be reached with less layers with more nodes each however this is more processor intensive than using more smaller layers.
"""

learning_rate = 0.01
training_epochs = 15
batch_size = 256
display_step = 1
n_inputs = 784
n_h1 = 256
n_h2 = 256
n_classes = 10

x = tf.placeholder(tf.float32, [None, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

def add_layer(inputs, input_size, output_size, relu=True):
    w = tf.Variable(tf.random_normal([input_size, output_size]))
    b = tf.Variable(tf.zeros([output_size]))
    
    layer = tf.add(tf.matmul(inputs, w), b)
    
    if relu:
        layer = tf.nn.relu(layer)
        
    return layer

fc1 = add_layer(x, n_inputs, n_h1)
fc2 = add_layer(fc1, n_h1, n_h2)
logits = add_layer(fc2, n_h2, n_classes, relu=False)

# Define loss and optimizer
"""
AdamOptimizer
It is just an algorithm for backpropagating that uses additional variables called slots that hold some additional information. For now it's not worth worrying about.
"""
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if epoch % display_step == 0:
            print ("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
    print ("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print ("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))