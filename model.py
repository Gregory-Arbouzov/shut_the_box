from box import Dice, Box, Player

import tensorflow as tf

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

#print(tf.compat.v1.convert_to_tensor(np.array(flip_options_for_model(6, box, player)).reshape((1, 12, 9))))
#print(tf.compat.v1.layers.placeholder(tf.float32, shape=(12,9)))

#print(np.array(flip_options_for_model(6, box, player)).shape)
#print(np.array(flip_options_for_model(6, box, player)).reshape((1, 12, 9)))

#inputs = keras.Input(shape=(12,9))
#x = keras.layers.Dense(9, activation='relu')(inputs)
#outputs = keras.layers.Dense(1, activation='sigmoid')(x)
#model = keras.Model(inputs=inputs, outputs=outputs)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(12, 9)),
  tf.keras.layers.Dense(9, activation = None),#activation='relu'),
  tf.keras.layers.Dense(9, activation='relu'),
  tf.keras.layers.Dense(1, activation = 'sigmoid')
])

# Create an instance of the model
model = model
"""
print(tf.compat.v1)

initializer = tf.compat.v1.variance_scaling_initializer()
# 2. Build the neural network
inputs =  tf.compat.v1.layers.placeholder(tf.float32, shape=(12,9))
x = tf.compat.v1.fully_connected(inputs, 9, activation_fn=tf.nn.relu,
weights_initializer=initializer)
outputs = tf.compat.v1.fully_connected(x, 1, activation_fn=None,
weights_initializer=initializer)
model = tf.nn.sigmoid(outputs)"""

#print(flip_options_for_model(6, box, player))

#model.summary()

input = tf.convert_to_tensor(np.array(flip_options_for_model(6, box, player)).reshape((1, 12, 9)))
#input = tf.compat.v1.convert_to_tensor(np.array(flip_options_for_model(6, box, player)).reshape((1, 12, 9)))
print(input[0])

output = model(input)
print(output)

print(tf.keras.ops.argmax(output, axis=1).numpy()[0])

#print(player.tile_flip_options(input_data[0][9], box))

def tile_flip_decision(input, output):
    decision = tf.argmax(output, axis=1).numpy()[0]
    return input[0][decision]

print(tile_flip_decision(input, output))

#print(box.tiles)

model_flip_box_update(tile_flip_decision(input, output), box)

print(player.get_current_score(box))



y = 1.0 - float(tf.reduce_max(output, axis=1).numpy()[0])
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
    all_discounted_rewards = [discount_rewards(rewards, discount_rate)
    for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std() + 1e-8
    return [(discounted_rewards - reward_mean)/reward_std
    for discounted_rewards in all_discounted_rewards]


discount_rate = 0.99



print(type(model.layers[1]))

learning_rate = 0.01

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

n_iterations = 250 # number of training iterations
n_max_steps = 1000 # max steps per episode
n_games_per_update = 10 # train the policy every 10 episodes
save_iterations = 10 # save the model every 10 training iterations
discount_rate = 0.95

checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
checkpoint_manager = tf.train.CheckpointManager(
    checkpoint, "./my_policy_net_pg_tf2", max_to_keep=3
)

for iteration in range(n_iterations):
    all_rewards = []  # all sequences of raw rewards for each episode
    all_gradients = []  # gradients saved at each step of each episode
    for game in range(n_games_per_update):
        current_rewards = []  # all raw rewards from the current episode
        current_gradients = []  # gradients from the current episode

        first_dice = Dice()
        second_dice = Dice()
        box = Box()
        player = Player(box, first_dice, second_dice)

        for step in range(9):
            roll_total = first_dice.roll()[0] + second_dice.roll()[0]
            done = len(player.tile_flip_options(roll_total, box)) == 0
            if done:
                break

            model_input = tf.convert_to_tensor(
                np.array(flip_options_for_model(roll_total, box, player)).reshape((1, 12, 9)),
                dtype=tf.float32,
            )

            with tf.GradientTape() as tape:
                output = model(model_input, training=True)
                action_idx = tf.argmax(output, axis=1)
                action_score = tf.reduce_max(output, axis=1)
                y_true = tf.ones_like(action_score)
                loss = tf.keras.losses.binary_crossentropy(y_true, action_score)
                loss = tf.reduce_mean(loss)

            gradients = tape.gradient(loss, model.trainable_variables)
            gradients = [
                tf.zeros_like(var) if grad is None else grad
                for grad, var in zip(gradients, model.trainable_variables)
            ]

            selected_flip = model_input[0][int(action_idx.numpy()[0])]
            model_flip_box_update(selected_flip.numpy(), box)

            #print(player.get_current_score(box))

            reward = 123456789 - player.get_current_score(box)
            current_rewards.append(float(reward))
            current_gradients.append(gradients)

        all_rewards.append(current_rewards)
        all_gradients.append(current_gradients)

    all_rewards = discount_and_normalize_rewards(all_rewards, discount_rate)

    mean_gradients = []
    for var_index in range(len(model.trainable_variables)):
        weighted_grads = []
        for game_index, rewards in enumerate(all_rewards):
            for step, reward in enumerate(rewards):
                weighted_grads.append(reward * all_gradients[game_index][step][var_index])

        if weighted_grads:
            grad_tensor = tf.reduce_mean(tf.stack(weighted_grads, axis=0), axis=0)
        else:
            grad_tensor = tf.zeros_like(model.trainable_variables[var_index])
        mean_gradients.append(grad_tensor)

    optimizer.apply_gradients(zip(mean_gradients, model.trainable_variables))

    if iteration % save_iterations == 0:
        checkpoint_manager.save()


#optimizer = keras.optimizers.Adam(learning_rate)

#loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

#grads = tf.tape.gradient(loss_value, model.trainable_weights)"""








if __name__ == "__main__":
    pass