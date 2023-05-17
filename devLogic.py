import random

class Card:
    def __init__(self, card_type):
        self.type = card_type

class Deck:
    def __init__(self):
        self.cards = []
        self.initialize_deck()

    def initialize_deck(self):
        types = ['Knight'] * 14 + ['Road Building'] * 2 + ['Year of Plenty'] * 2 + ['Monopoly'] * 2 + ['Victory Point'] * 5
        for card_type in types:
            card = Card(card_type)
            self.cards.append(card)

    def shuffle(self):
        random.shuffle(self.cards)

    def draw_card(self):
        if len(self.cards) > 0:
            return self.cards.pop()
        else:
            return None

    def cards_remaining(self):
        return len(self.cards)
