from box import Dice, Box, Player
    

first_dice = Dice()
second_dice = Dice()
#print(first_dice.roll())

box = Box()
print(box.score())

player = Player(box, first_dice, second_dice)
print(player.roll_dice(first_dice, second_dice))
print(player.tile_flip_options(box))