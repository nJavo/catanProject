import random


class Tile:
    def __init__(self, resource, number):
        self.resource = resource
        self.number = number

def create_board():
    numbers = [5, 2, 6, 3, 8, 10, 9, 12, 11, 4, 8, 10, 9, 4, 5, 6, 3, 11]

    resources = ['Brick', 'Wood', 'Sheep', 'Wheat', 'Ore'] * 3 + ['Wood', 'Sheep', 'Wheat']
    resources.append('Desert')
    random.shuffle(resources)

    # resources = ['Wheat', 'Ore', 'Ore', 'Wood', 'Sheep', 'Desert', 'Wood', 'Sheep', 'Sheep', 'Sheep', 'Wood', 'Brick', 'Wheat', 'Ore', 'Brick', 'Wood', 'Wheat', 'Brick', 'Wheat']
    # startNumber = 10
    
    startNumber = random.randint(0, 11)

    # print(resources)
    # print(startNumber)

    ring_1 = [None] * 12
    ring_2 = [None] * 6
    ring_3 = [None]

    last_placed_index = None

    for i in range(12):
        resource = resources[(startNumber + i) % 12]
        hexagon_index = (startNumber + i) % 12

        if resource != 'Desert':
            ring_1[hexagon_index] = numbers.pop(0)
            last_placed_index = hexagon_index

    last_placed_index_2 = last_placed_index

    for i in range(6):
        hexagon_index = 12 + ((last_placed_index + i + 1) % 6)
        resource = resources[hexagon_index]

        if resource != 'Desert':
            ring_2[((last_placed_index + i + 1) % 6)] = numbers.pop(0)
            last_placed_index_2 = hexagon_index - 12

    resource = resources[18]
    if resource != 'Desert':
        ring_3[0] = numbers.pop(0)

    number_placements = ring_1 + ring_2 + ring_3        

    #print(number_placements)
    
    board = [Tile(resource, number) for resource, number in zip(resources, number_placements)]      

    return board

def print_board(board):
    for tile in board:
        print(f'Resource: {tile.resource}, Number: {tile.number}')