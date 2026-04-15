from box import Dice, Box, Player

import tensorflow as tf


#from tensorflow import keras

#from keras.layers import Dense

import tensorflow as tf
print("TensorFlow version:", tf.__version__)

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

import numpy as np

first_dice = Dice()
second_dice = Dice()

box = Box()

player = Player(box, first_dice, second_dice)

def box_tiles_for_model(box):
    box_rep = [0,0,0,0,0,0,0,0,0]
    
    for i in range(9):
        if box.tiles[i] == i + 1:
            box_rep[i] = 1
        else:
            box_rep[i] = 0
    
    return box_rep

def flip_options_for_model(dice_sum, box, player):
    flip_options = player.tile_flip_options(dice_sum, box)
    flip_reps = []

    for option in flip_options:
        box_rep = box_tiles_for_model(box)
        for i in range(len(option)):
            box_rep[option[i] - 1] = 0
        flip_reps.append(box_rep)
    
    flip_reps.extend([box_tiles_for_model(box)] * (12 - len(flip_reps))) 
    
    return flip_reps

def model_flip_box_update(model_flips, box):
    for i in range(len(box.tiles)):
        if np.array(model_flips)[i] == 0:
            box.tiles[i] = ""
        else:
            pass
    
    return box


#print(np.array(flip_options_for_model(6, box, player)).shape)
#print(np.array(flip_options_for_model(6, box, player)).reshape((1, 12, 9)))

#inputs = keras.Input(shape=(12,9))
#x = keras.layers.Dense(9, activation='relu')(inputs)
#outputs = keras.layers.Dense(1, activation='sigmoid')(x)
#model = keras.Model(inputs=inputs, outputs=outputs)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(12, 9)),
  tf.keras.layers.Dense(9, activation='relu'),
  tf.keras.layers.Dense(1, activation = 'sigmoid')
])

# Create an instance of the model
model = model

old="""
initializer = tf.compat.v1.variance_scaling_initializer()
# 2. Build the neural network
inputs =  tf.compat.v1.layers.dense.placeholder(tf.float32, shape=(12,9))
x = tf.fully_connected(inputs, 9, activation_fn=tf.nn.relu,
weights_initializer=initializer)
outputs = tf.compat.v1.fully_connected(x, 1, activation_fn=None,
weights_initializer=initializer)
model = tf.nn.sigmoid(outputs)"""

#print(flip_options_for_model(6, box, player))

#model.summary()

input = tf.keras.ops.convert_to_tensor(np.array(flip_options_for_model(6, box, player)).reshape((1, 12, 9)))
print(input[0])

output = model(input)
print(output)

print(tf.keras.ops.argmax(output, axis=1).numpy()[0])

#print(player.tile_flip_options(input_data[0][9], box))

def tile_flip_decision(input, output):
    decision = tf.keras.ops.argmax(output, axis=1).numpy()[0]
    return input[0][decision]

print(tile_flip_decision(input, output))

#print(box.tiles)

model_flip_box_update(tile_flip_decision(input, output), box)

print(player.get_current_score(box))



y = 1 - tf.keras.ops.max(output, axis=1).numpy()[0]
print(y)

print(box.tiles)
print(player.get_current_score(box))
#print(first_dice.roll())
print(len(player.tile_flip_options(first_dice.roll()[0] + second_dice.roll()[0], box)) > 0)

#obs, reward, done, info = env.step(action)

def discount_rewards(rewards, discount_rate):
    discounted_rewards = np.empty(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards

def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards)
    for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean)/reward_std
    for discounted_rewards in all_discounted_rewards]


discount_rate = 0.99



print(type(model.layers[1]))

learning_rate = 0.01

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=model.layers[1])

optimizer = tf.train.AdamOptimizer(learning_rate)

grads_and_vars = optimizer.compute_gradients(cross_entropy)
gradients = [grad for grad, variable in grads_and_vars]
gradient_placeholders = []
grads_and_vars_feed = []

for grad, variable in grads_and_vars:
    gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
    gradient_placeholders.append(gradient_placeholder)
    grads_and_vars_feed.append((gradient_placeholder, variable))
    
training_op = optimizer.apply_gradients(grads_and_vars_feed)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_iterations = 250 # number of training iterations
n_max_steps = 1000 # max steps per episode
n_games_per_update = 10 # train the policy every 10 episodes
save_iterations = 10 # save the model every 10 training iterations
discount_rate = 0.95

with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        all_rewards = [] # all sequences of raw rewards for each episode
        all_gradients = [] # gradients saved at each step of each episode
        for game in range(n_games_per_update):
            current_rewards = [] # all raw rewards from the current episode
            current_gradients = [] # all gradients from the current episode

            
            first_dice = Dice()
            second_dice = Dice()

            box = Box()

            player = Player(box, first_dice, second_dice)

            for step in range(9):

                obs = box.tiles
                reward = 123456789 - player.get_current_score(box)
                roll_total = first_dice.roll()[0] + second_dice.roll()[0]
                done = len(player.tile_flip_options(roll_total, box)) == 0

                input = keras.ops.convert_to_tensor(np.array(flip_options_for_model(roll_total, box, player)).reshape((1, 12, 9)))

                output = model(input)

                #print(tile_flip_decision(input, output))

                action_val, gradients_val = sess.run(
                        [tile_flip_decision(input, output), gradients],
                        feed_dict={inputs: obs.reshape(1, 9)}) # one obs
                
                #obs, reward, done, info = env.step(action_val[0][0])
                current_rewards.append(1 * reward)
                current_gradients.append(gradients_val)
                if done:
                    break
            all_rewards.append(current_rewards)
            all_gradients.append(current_gradients)

        # At this point we have run the policy for 10 episodes, and we are
        # ready for a policy update using the algorithm described earlier.
        all_rewards = discount_and_normalize_rewards(all_rewards)
        feed_dict = {}
        for var_index, grad_placeholder in enumerate(gradient_placeholders):
            # multiply the gradients by the action scores, and compute the mean
            mean_gradients = np.mean(
                [reward * all_gradients[game_index][step][var_index]
            for game_index, rewards in enumerate(all_rewards)
            for step, reward in enumerate(rewards)],
 axis=0)
        feed_dict[grad_placeholder] = mean_gradients
    sess.run(training_op, feed_dict=feed_dict)
    if iteration % save_iterations == 0:
        saver.save(sess, "./my_policy_net_pg.ckpt")


#optimizer = keras.optimizers.Adam(learning_rate)

#loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

#grads = tf.tape.gradient(loss_value, model.trainable_weights)"""








if __name__ == "__main__":
    pass