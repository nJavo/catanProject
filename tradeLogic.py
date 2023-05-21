from playerInventory import PlayerInventory

class Trade:
    def __init__(self, players):
        self.players = players
        self.proposed_trades = {player: {} for player in players}
        self.accepted_trade = None

    def propose_trade(self, offering_player, receiving_players, offering_cards, receiving_cards):
        for receiving_player in receiving_players:
            if receiving_player in self.players:
                self.proposed_trades[offering_player][receiving_player] = (offering_cards, receiving_cards)
                print(f"Player {offering_player} has offered a trade to Player {receiving_player}.")
                print(f"Offering cards: {offering_cards}")
                print(f"Receiving cards: {receiving_cards}")
            else:
                print("Invalid player(s) for trade.")

    def edit_trade(self, offering_player, receiving_player, edited_offering_cards, edited_receiving_cards):
        if offering_player in self.players and receiving_player in self.players:
            if offering_player in self.proposed_trades and receiving_player in self.proposed_trades[offering_player]:
                self.proposed_trades[offering_player][receiving_player] = (edited_offering_cards, edited_receiving_cards)
                print(f"Player {offering_player} has edited the trade with Player {receiving_player}.")
                print(f"Offering cards: {edited_offering_cards}")
                print(f"Receiving cards: {edited_receiving_cards}")
            else:
                print(f"No existing trade between Player {offering_player} and Player {receiving_player}.")
        else:
            print("Invalid player(s) for trade.")

    def accept_trade(self, receiving_player):
        if receiving_player in self.players:
            if self.accepted_trade is None:
                for offering_player in self.proposed_trades:
                    if receiving_player in self.proposed_trades[offering_player]:
                        self.accepted_trade = (offering_player, receiving_player)
                        print(f"Player {receiving_player} has accepted the trade offered by Player {offering_player}.")
                        break
                else:
                    print("No existing trade involving the receiving player.")
            else:
                print("A trade has already been accepted.")
        else:
            print("Invalid player for trade.")

    def reject_trade(self, offering_player, receiving_player):
        if offering_player in self.players and receiving_player in self.players:
            if offering_player in self.proposed_trades and receiving_player in self.proposed_trades[offering_player]:
                del self.proposed_trades[offering_player][receiving_player]
                print(f"Player {receiving_player} has rejected the trade offered by Player {offering_player}.")
            else:
                print(f"No existing trade between Player {offering_player} and Player {receiving_player}.")
        else:
            print("Invalid player(s) for trade.")
