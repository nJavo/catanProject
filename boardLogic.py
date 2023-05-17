import random


class Tile:
    def __init__(self, resource, number):
        self.resource = resource
        self.number = number

def create_board():
    resources = ['Brick', 'Wood', 'Sheep', 'Wheat', 'Ore'] * 3 + ['Wood', 'Sheep', 'Wheat']
    numbers = [5, 2, 6, 3, 8, 10, 9, 12, 11, 4, 8, 10, 9, 4, 5, 6, 3, 11]

    # Add the desert tile
    resources.append('Desert')

    random.shuffle(resources)

    startNumber = random.randint(0, 11)
    
    for 

    board = [Tile(resource, number) for resource, number in zip(resources, numbers)]

    return board

def print_board(board):
    for tile in board:
        print(f'Resource: {tile.resource}, Number: {tile.number}')