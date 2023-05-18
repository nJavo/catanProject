import random

def roll_dice(board, settlement_spots, players):
    dice_result1 = random.randint(1, 6)
    dice_result2 = random.randint(1, 6)
    dice_sum = dice_result1 + dice_result2

    # Get tiles that match the dice roll
    matching_tiles = [tile for tile in board if tile.number == dice_sum]

    # For each spot
    for spot in settlement_spots:
        # If the spot has a building
        if spot.building is not None:
            # Check if the spot is connected to a matching tile
            connected_matching_tiles = [tile for tile in spot.connected_tiles if tile in matching_tiles]
            if connected_matching_tiles:
                # Determine the player who owns the building
                player_id = spot.building['player']
                # Get the player's inventory
                player = next(player for player in players if player.player_id == player_id)
                # Distribute resources for each matching tile
                for tile in connected_matching_tiles:
                    resource_type = tile.resource.lower()
                    player.add_resource_cards(resource_type, 1)
                    print(f'Player {player_id} received 1 {resource_type} from tile {board.index(tile)}.')

    return dice_sum