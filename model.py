from box import Dice, Box, Player

import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

first_dice = Dice()
second_dice = Dice()

box = Box()
    #print(box.score())

player = Player(box, first_dice, second_dice)

# 1. Specify the neural network architecture
n_inputs = 10 # == env.observation_space.shape[0]
n_hidden = 10 # it's a simple task, we don't need more hidden neurons
n_outputs = 9 # only outputs the probability of accelerating left
initializer = tf.contrib.layers.variance_scaling_initializer()
# 2. Build the neural network
X = tf.placeholder(tf.float32, shape=[None, n_inputs])
hidden = fully_connected(X, n_hidden, activation_fn=tf.nn.elu,
 weights_initializer=initializer)
logits = fully_connected(hidden, n_outputs, activation_fn=None,
 weights_initializer=initializer)
outputs = tf.nn.softmax(logits)
# 3. Select a random action based on the estimated probabilities
tile_probs = tf.concat(axis=1, values=[outputs, 1 - outputs])
action = null
init = tf.global_variables_initializer()

if __name__ == "__main__":
    print('yes')