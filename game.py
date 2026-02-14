from box import Dice, Box, Player

def play():  

    print()
    print()
    print('STARTING A NEW SHUT-THE-BOX GAME!')

    first_dice = Dice()
    second_dice = Dice()
    #print(first_dice.roll())

    box = Box()
    #print(box.score())

    player = Player(box, first_dice, second_dice)
    #print(player.roll_dice(first_dice, second_dice))

    game_over = False

    while not game_over:
        print()
        print('The Current Tiles are: ' + str(box.tiles))
        dice_sum = player.roll_dice(first_dice, second_dice)
        print('The Dice Sum to: ' + str(dice_sum))
        print('Your Flip Options are: ' + str(player.tile_flip_options(dice_sum, box)))

        if player.tile_flip_options(dice_sum, box) == []:
            print()
            print('Game Over.... Your Final Score is: ' + str(player.get_current_score(box)))
            print()
            game_over = True

        old_score = box.score()

        while old_score == box.score() and not game_over:
            print()
            tile_selection = input("Select a Group of Tiles to Flip: ")
            tile_select_list = eval(tile_selection)
            player.tile_flip_choice(tile_select_list, dice_sum, box)
            
        if not game_over:
            print()
            print('Your Current Score is: ' + str(player.get_current_score(box)))
            print()
            print("Let's Roll Again!!")
            print()



