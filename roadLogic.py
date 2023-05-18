from road_initialData import initial_data_roads

class RoadSpot:
    def __init__(self, road=None):
        self.road = road
        self.neighboring_roads = []
        self.neighboring_spots = []
        self.building = None

class RoadLogic:    
    @staticmethod
    def setup_board(board, road):
    # Create the list of RoadSpot instances
        road_spots = [RoadSpot() for _ in range(72)]

        # For each RoadSpot, use the initial data to assign its attributes
        for i, road_data in enumerate(initial_data_roads):
            road = road_spots[i]
            road.neighboring_spots = [road_spots[j] for j in road_data['neighboring_spots']]
            road.neighboring_roads = [road_spots[j] for j in road_data['neighboring_roads']]

        return road_spots

    @staticmethod
    def is_road_available(spot):
        if road.building is not None:
            return False
        for neighbor in spot.neighboring_spots:
            if neighbor.building is not None:
                return True
                #bool1 = False
        for neighbor in spot.neighboring_roads:
            if neighbor.building is not None:
                bool2 = True     
        # return bool1 || bool2
        return True
        
    @staticmethod    
    def print_spots(road_spots, board):
        for i, spot in enumerate(road_spots):
            neighboring_spot_indices = [road_spots.index(neighbor) for neighbor in spot.neighboring_spots]
            neighboring_road_indices = [road_spots.index(neighbor) for neighbor in spot.neighboring_roads]
            print(f'Spot {i}:')
            print(f'  Neighboring spots: {neighboring_spot_indices}')
            print(f'  Neighboring road: {neighboring_road_indices}')
            print(f'  Port: {spot.road}')
            print(f'  Building: {spot.building}\n')

    @staticmethod
    def build_road(road_spots, road):
        if RoadLogic.is_road_available(road):
            road.building = 'Road'
            print(f'A road has been built on {road_spots.index(road)}.')
        else:
            print('This spot is not available for building.')
