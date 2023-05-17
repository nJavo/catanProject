import random

class Port:
    def __init__(self, port_type):
        self.type = port_type

class Ports:
    def __init__(self):
        self.ports = []
        self.initialize_deck()

    def initialize_deck(self):
        types = ['Brick', 'Wood', 'Sheep', 'Wheat', 'Ore'] + ['Generic']*4
        for port_type in types:
            port = Port(port_type)
            self.ports.append(port)

    def shuffle(self):
        random.shuffle(self.ports)
