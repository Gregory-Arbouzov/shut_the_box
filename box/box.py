import random

from .helper import get_distinct_sums

class Dice:
    def __init__(self):
        self.sides = (1, 2, 3, 4, 5, 6)
    
    def roll(self):
        return random.sample(self.sides, 1)
    
class Box:
    def __init__(self):
        self.tiles = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    def flip_tiles(self, tiles_to_flip_nums):
        for tile in tiles_to_flip_nums:
            self.tiles[tile - 1] = ""
    
    def score(self):
        score_str = ""
        for tile in self.tiles:
            if tile != "":
                score_str = score_str + str(tile)

        return int(score_str)

class Player:
    def __init__(self, box_instance, dice_instance1, dice_instance2):
        box_instance = Box()
        dice_instance1 = Dice()
        dice_instance2 = Dice()

    def get_current_score(self, box_instance):
        return box_instance.score()
    
    def roll_dice(self, dice_instance1, dice_instance2):
        return sum(dice_instance1.roll() + dice_instance2.roll())
    
    def tile_flip_options(self, roll_total, box_instance):
        options_list = get_distinct_sums(roll_total)
        
        for option in options_list:
            for tile in option:
                if box_instance.tiles[tile - 1] == "":
                    options_list = [other_options for other_options in options_list if other_options != option].copy()
        return options_list 
    
    def tile_flip_choice(self, tiles_to_flip, roll_total, box_instance):
        if tiles_to_flip in self.tile_flip_options(roll_total, box_instance):
            for tile in tiles_to_flip:
                box_instance.tiles[tile - 1] = ""
        else:
            print('Invalid Combination of Tiles Selected')
        return box_instance.tiles