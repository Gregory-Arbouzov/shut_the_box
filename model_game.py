import numpy as np
import tensorflow as tf
from box import Dice, Box, Player

import random
import math
import statistics

def build_model():
    return tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(12, 9)),
        tf.keras.layers.Dense(9, activation=None),
        tf.keras.layers.Dense(9, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])

def box_tiles_for_model(box):
    return [1 if box.tiles[i] == i + 1 else 0 for i in range(9)]

def flip_options_for_model(dice_sum, box, player):
    flip_options = player.tile_flip_options(dice_sum, box)
    flip_reps = []
    for option in flip_options:
        box_rep = box_tiles_for_model(box)
        for tile in option:
            box_rep[tile - 1] = 0
        flip_reps.append(box_rep)
    flip_reps.extend([box_tiles_for_model(box)] * (12 - len(flip_reps)))
    return flip_reps

def random_flip(dice_sum, box, player):
    flip_options = player.tile_flip_options(dice_sum, box)
    random_index = random.randrange(len(flip_options))

    return flip_options[random_index]

def max_product_flip(dice_sum, box, player):
    flip_options = player.tile_flip_options(dice_sum, box)
    flip_products = [math.prod(option) for option in flip_options]
    max_product_index = flip_products.index(max(flip_products))

    return flip_options[max_product_index]

def model_flip_box_update(model_flips, box):
    for i in range(len(box.tiles)):
        if np.array(model_flips)[i] == 0:
            box.tiles[i] = ""

def model_play():
    model = build_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)  # only needed for checkpoint restore
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(ckpt, "./my_policy_net_pg_tf2", max_to_keep=3)
    latest = manager.latest_checkpoint
    if latest is None:
        raise FileNotFoundError("No checkpoint found in ./my_policy_net_pg_tf2")
    ckpt.restore(latest).expect_partial()
    print("Loaded:", latest)
    first_dice = Dice()
    second_dice = Dice()
    box = Box()
    player = Player(box, first_dice, second_dice)

    print()
    print()
    print('STARTING A NEW SHUT-THE-BOX GAME!')

    for _ in range(9):
        print()
        print('The Current Tiles are: ' + str(box.tiles))
        roll_total = first_dice.roll()[0] + second_dice.roll()[0]
        print('The Dice Sum to: ' + str(roll_total))
        options = player.tile_flip_options(roll_total, box)
        print('Agent Flip Options are: ' + str(options))
        if len(options) == 0:
            print()
            print('Game Over.... Agent Final Score is: ' + str(player.get_current_score(box)))
            print()
            break
        model_input = tf.convert_to_tensor(
            np.array(flip_options_for_model(roll_total, box, player)).reshape((1, 12, 9)),
            dtype=tf.float32,
        )

        output = model(model_input, training=False)
        choice_idx = int(tf.argmax(output, axis=1).numpy()[0])
        selected_flip = model_input[0][choice_idx].numpy()
        #print('Model Selected Flip: ', selected_flip)
        model_flip_box_update(selected_flip, box)

        print()
        print('Agent Current Score is: ' + str(player.get_current_score(box)))
        print()
        print("Let's Roll Again!!")
        print()

def model_play_for_batch():
    model = build_model()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)  # only needed for checkpoint restore
    ckpt = tf.train.Checkpoint(model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(ckpt, "policies/my_policy_net_pg_tf2", max_to_keep=3)
    latest = manager.latest_checkpoint
    if latest is None:
        raise FileNotFoundError("No checkpoint found in policies/my_policy_net_pg_tf2")
    ckpt.restore(latest).expect_partial()
    first_dice = Dice()
    second_dice = Dice()
    box = Box()
    player = Player(box, first_dice, second_dice)

    for _ in range(9):
        roll_total = first_dice.roll()[0] + second_dice.roll()[0]
        options = player.tile_flip_options(roll_total, box)
        if len(options) == 0:
            break

        model_input = tf.convert_to_tensor(
            np.array(flip_options_for_model(roll_total, box, player)).reshape((1, 12, 9)),
            dtype=tf.float32,
        )
        
        output = model(model_input, training=False)
        choice_idx = int(tf.argmax(output, axis=1).numpy()[0])
        selected_flip = model_input[0][choice_idx].numpy()
        model_flip_box_update(selected_flip, box)
    
    return player.get_current_score(box)

def coded_policy_play(policy):  
    first_dice = Dice()
    second_dice = Dice()
    
    box = Box()

    player = Player(box, first_dice, second_dice)

    game_over = False

    while not game_over:
        dice_sum = player.roll_dice(first_dice, second_dice)
        
        if player.tile_flip_options(dice_sum, box) == []:
            game_over = True

        old_score = box.score()

        while old_score == box.score() and not game_over:    
            player.tile_flip_choice(policy(dice_sum, box, player), dice_sum, box)
            #random_flip(dice_sum, box, player)
            
    return player.get_current_score(box)   

if __name__ == "__main__":
    random_model_scores = [coded_policy_play(random_flip) for _ in range(10000)]
    max_product_flip_model_scores = [coded_policy_play(max_product_flip) for _ in range(10000)]
    agent_model_scores = [model_play_for_batch() for _ in range(1000)]

    print("Agent Average Score: " + str(statistics.mean(agent_model_scores)))
    print("Max Product Strategy Average Score: " + str(statistics.mean(max_product_flip_model_scores)))
    print("Random Flip Average Score: " + str(statistics.mean(random_model_scores)))