from box import Dice, Box, Player
    

first_dice = Dice()
second_dice = Dice()
#print(first_dice.roll())

box = Box()
print(box.score())

player = Player(box, first_dice, second_dice)
#print(player.roll_dice(first_dice, second_dice))

dice_sum = player.roll_dice(first_dice, second_dice)

print('The Dice Sum to: ' + str(dice_sum))

print('The Current Tiles are: ' + str(box.tiles))

print(player.tile_flip_options(dice_sum, box))
