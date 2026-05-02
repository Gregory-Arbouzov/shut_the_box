# play_with_model.py
import numpy as np
import tensorflow as tf
from box import Dice, Box, Player
# ---- Must match training architecture exactly ----

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
def model_flip_box_update(model_flips, box):
    for i in range(len(box.tiles)):
        if np.array(model_flips)[i] == 0:
            box.tiles[i] = ""
# ---- Load checkpoint ----

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
    # ---- Use model to play one game ----
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

if __name__ == "__main__":
    model_play()