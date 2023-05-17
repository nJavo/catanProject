class PlayerInventory:
    def __init__(self, player_id):
        self.player_id = player_id
        self.development_cards = []
        self.resource_cards = {"brick": 0, "wood": 0, "ore": 0, "wheat": 0, "sheep": 0}
        self.victory_points = 0
        self.roads = 15
        self.cities = 4
        self.settlements = 5

    def add_development_card(self, card):
        self.development_cards.append(card)

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

    def place_settlement(self):
        if self.settlements > 0:
            self.settlements -= 1