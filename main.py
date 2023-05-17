import boardLogic
import devLogic
import settlementLogic

def main():
    board = boardLogic.create_board()
    boardLogic.print_board(board)

    # Set up the settlement spots
    ports = ['Brick', 'Wood', 'Sheep', 'Wheat', 'Ore', 'Generic', 'Generic', 'Generic', 'Generic']  # You need to define this list
    settlement_spots = settlementLogic.SettlementLogic.setup_board(board, ports)

    deck = devLogic.Deck()
    deck.shuffle()

    while True:
        print("\nActions:")
        print("1. Draw card")
        print("2. Check remaining cards")
        print("3. Check if a settlement spot is available")
        print("4. Exit")
        user_input = input("Choose an action: ")

        if user_input == "1":
            card = deck.draw_card()
            if card is not None:
                print(f"You drew a {card.type} card.")
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
            break
        else:
            print("Invalid input. Please enter a number between 1 and 4.")

if __name__ == "__main__":
    main()