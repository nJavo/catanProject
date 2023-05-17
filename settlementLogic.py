from settlement_initialData import initial_data

class SettlementSpot:
    def __init__(self, port=None):
        self.port = port
        self.connected_tiles = []
        self.neighboring_spots = []
        self.building = None

class SettlementLogic:    
    @staticmethod
    def setup_board(board, ports):
    # Create the list of SettlementSpot instances
        settlement_spots = [SettlementSpot() for _ in range(54)]

        # For each SettlementSpot, use the initial data to assign its attributes
        for i, spot_data in enumerate(initial_data):
            spot = settlement_spots[i]
            spot.connected_tiles = [board[j] for j in spot_data['connected_tiles']]
            spot.neighboring_spots = [settlement_spots[j] for j in spot_data['neighboring_spots']]
            if spot_data['port'] != None:
                spot.port = ports[spot_data['port']]

        return settlement_spots

    @staticmethod
    def is_spot_available(spot):
        if spot.building is not None:
            return False
        for neighbor in spot.neighboring_spots:
            if neighbor.building is not None:
                return False
        return True
    
    @staticmethod
    def print_spots(settlement_spots, board):
        for i, spot in enumerate(settlement_spots):
            connected_tile_indices = [board.index(tile) for tile in spot.connected_tiles]
            neighboring_spot_indices = [settlement_spots.index(neighbor) for neighbor in spot.neighboring_spots]
            print(f'Spot {i}:')
            print(f'  Connected tiles: {connected_tile_indices}')
            print(f'  Neighboring spots: {neighboring_spot_indices}')
            print(f'  Port: {spot.port}')
            print(f'  Building: {spot.building}\n')

    @staticmethod
    def build_settlement(settlement_spots, spot):
        if SettlementLogic.is_spot_available(spot):
            spot.building = 'Settlement'
            print(f'A settlement has been built on spot {settlement_spots.index(spot)}.')
        else:
            print('This spot is not available for building.')