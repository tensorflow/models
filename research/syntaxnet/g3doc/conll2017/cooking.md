# Human Readable Neural Net Recipe

An old-fashioned, hand-crafted, CPU-to-table recipe for the CoNLL2017 Shared
Task.

## Ingredients

*   3 layer-normalized LSTM networks
*   1 layer-normalized feed-forward cell
*   1 treebank for training
*   1 treebank for tuning
*   1 DRAGNN framework cooking dish (non-stick)
*   1 TensorFlow™ non-convex, stochastic oven (ADAM configuration)
*   Generous helpings of CPU time

## Directions

_First, assemble your network inside your DRAGNN dish:_

*   Put the first LSTM directly on top of your treebank's characters. Use a
    spatula to mark the word boundaries for the rest of the model.
*   Next, lay the second LSTM backwards across the top of the first. Don't
    forget to project the hidden layer down by 1/4 with each LSTM you add to the
    pan. Remember, you only need to put an LSTM cell down just once for each
    word you marked in the previous step.
*   Now lay down your final LSTM running in the forwards direction. Sprinkle
    with POS tags at a ratio of 1:8.
*   Finally, gently put the feed-forward cell down on top. Carefully braid the
    three inputs of the feed-forward layer into the topmost two LSTMs in the
    pan. You'll need to follow the dependency arcs laid out in your training
    treebank to get it just right. (Don't worry -- once baked, your model will
    be able to do this part itself.)
*   Use a good pair of L2-norm scissors to clip any gradients you find that have
    greater than unit norm. (Your gradients should all fit inside the L2 unit
    ball.)
*   Always double check your work before baking!

_Baking instructions:_

*   Prepare your TensorFlow™ brand ADAM-style oven: set temperature to
    beta1=0.9, beta2=0.9, eps=0.0001, and learning rate of 0.001.
*   Par-bake your LSTM crust: Place your assembled dish inside your oven. Bake
    only with POS tags for 10K iterations.
*   Finish cooking the top layers: Next bake for at least 100-200K iterations,
    up to about ~900K iterations for parsing.
*   To speed up convergence, group your training examples into minibatches of
    size 4, and put 4 of them in the oven at a time.
*   Pull your dish out of the oven when you notice holdout set accuracy is no
    longer increasing. Your parser is now ready to serve!

## General cooking tips

Sometimes your dish just doesn't come out right. If this happens to you, just
change your oven's random seed and try again. Usually just one or two restarts
is enough.
