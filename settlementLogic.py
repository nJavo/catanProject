from initialData import initial_data

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