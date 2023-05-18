import boardLogic
import devLogic
import settlementLogic
from playerInventory import PlayerInventory
import logicRoll

def main():
    num_players = 4

    board = boardLogic.create_board()
    boardLogic.print_board(board)

    ports = ['Brick', 'Wood', 'Sheep', 'Wheat', 'Ore', 'Generic', 'Generic', 'Generic', 'Generic']
    settlement_spots = settlementLogic.SettlementLogic.setup_board(board, ports)

    deck = devLogic.Deck()
    deck.shuffle()

    player_inventories = [PlayerInventory(player_id + 1) for player_id in range(num_players)]

    current_player_index = 0

    while True:
        current_player = player_inventories[current_player_index]
        print(f"\nPlayer {current_player.player_id}'s turn:")

        print("Actions:")
        print("1. Draw card")
        print("2. Check remaining cards")
        print("3. Check if a settlement spot is available")
        print("4. Print settlement spots")
        print("5. Build settlement")
        print("6. Roll the dice")
        print("7. End turn")
        print("8. Exit game")
        print("9. View your inventory")
        user_input = input("Choose an action: ")

        if user_input == "1":
            card = deck.draw_card()
            if card is not None:
                print(f"You drew a {card.type} card.")
                current_player.add_development_card(card.type)
            else:
                print("No more cards in the deck.")
        elif user_input == "2":
            print(f"There are {deck.cards_remaining()} cards remaining in the deck.")
        elif user_input == "3":
            spot_number = int(input("Enter the number of the settlement spot to check: "))
            spot = settlement_spots[spot_number]
            if settlementLogic.SettlementLogic.is_spot_available(spot):
                print("This spot is available for building.")
            else:
                print("This spot is not available for building.")
        elif user_input == "4":
            settlementLogic.SettlementLogic.print_spots(settlement_spots, board)
        elif user_input == "5":
            spot_number = int(input("Enter the number of the settlement spot to build on: "))
            spot = settlement_spots[spot_number]
            settlementLogic.SettlementLogic.build_settlement(settlement_spots, spot, current_player)
        elif user_input == "6":
            dice_result = logicRoll.roll_dice(board, settlement_spots, player_inventories)
            print(f"The dice roll result was {dice_result}")
        elif user_input == "7":
            current_player_index = (current_player_index + 1) % num_players
        elif user_input == "8":
            break
        elif user_input == "9":
            print(f"\nPlayer {current_player.player_id}'s Inventory:")
            print(f"Development Cards: {current_player.development_cards}")
            print(f"Resource Cards: {current_player.resource_cards}")
            print(f"Victory Points: {current_player.victory_points}")
            print(f"Roads: {current_player.roads}")
            print(f"Cities: {current_player.cities}")
            print(f"Settlements: {current_player.settlements}")
        else:
            print("Invalid input. Please enter a number between 1 and 9.")

if __name__ == "__main__":
    main()