import random

def roll_dice():
    dice_result1 = random.randint(1, 6)
    dice_result2 = random.randint(1, 6)
    dice_sum = dice_result1 + dice_result2
    return dice_sum
