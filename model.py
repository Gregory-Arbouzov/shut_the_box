from box import Dice, Box, Player

from tensorflow import keras as k

import numpy as np

first_dice = Dice()
second_dice = Dice()

box = Box()

player = Player(box, first_dice, second_dice)

inputs = k.Input(shape=(10,))
x = k.layers.Dense(10, activation='relu')(inputs)
outputs = k.layers.Dense(9, activation='softmax')(x)
model = k.Model(inputs=inputs, outputs=outputs)

#model.summary()

input_data = [[0, 2, 3, 4, 5, 6, 7, 8, 9, 4]]

output = model(k.ops.convert_to_tensor(input_data, dtype='float32'))

print(player.tile_flip_options(input_data[0][9], box))

def tile_flip_decision(input, output, box):
    player.tile_flip_options(box, input[9])
    anchor_tile = k.ops.argmax(output, axis=1).numpy()[0] - 1

print(output)

print(k.ops.argmax(output, axis=1).numpy())



"""
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
"""

if __name__ == "__main__":
    pass