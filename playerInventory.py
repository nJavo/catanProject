class PlayerInventory:
    def __init__(self, player_id):
        self.player_id = player_id
        self.development_cards = []
        self.resource_cards = {"brick": 0, "wood": 0, "ore": 0, "wheat": 0, "sheep": 0}
        self.victory_points = 0
        self.roads = 15
        self.cities = 4
        self.settlements = 5
        self.buildings = []  # To store settlement and city placements

    def add_development_card(self, card):
        self.development_cards.append(card)
        # Check if the card is a victory point and update points
        if card.type == 'Victory Point':
            self.increment_victory_points(1)

    def add_resource_cards(self, resource_type, amount):
        self.resource_cards[resource_type] += amount

    def remove_resource_cards(self, resource_type, amount):
        self.resource_cards[resource_type] = max(self.resource_cards[resource_type] - amount, 0)

    def increment_victory_points(self, points):
        self.victory_points += points

    def decrement_victory_points(self, points):
        self.victory_points = max(self.victory_points - points, 0)

    def place_road(self):
        if self.roads > 0:
            self.roads -= 1

    def place_city(self):
        if self.cities > 0:
            self.cities -= 1
            self.increment_victory_points(1)

    def place_settlement(self):
        if self.settlements > 0:
            self.settlements -= 1
            self.increment_victory_points(1)
            
    def add_building(self, building_type, spot_number):
        self.buildings.append({'type': building_type, 'spot': spot_number})