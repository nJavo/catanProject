class SettlementSpot:
    def __init__(self, port=None):
        self.port = port
        self.connected_tiles = []
        self.neighboring_spots = []
        self.building = None

class SettlementLogic:    
    def create_board_graph(board, ports):
        # Create an empty list for the settlement spots
        settlement_spots = [SettlementSpot() for _ in range(54)]

        settlement_spots[0].connected_tiles = [board[0], board[1], board[2]]
        settlement_spots[0].neighboring_spots = [settlement_spots[1], settlement_spots[2], settlement_spots[3]]
        settlement_spots[0].port = ports[0]


        return settlement_spots


    def is_spot_available(spot):
        if spot.building is not None:
            return False
        for neighbor in spot.neighboring_spots:
            if neighbor.building is not None:
                return False
        return True