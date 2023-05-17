import boardLogic
import devLogic
import settlementLogic



def main():
    board = boardLogic.create_board()
    boardLogic.print_board(board)

    deck = devLogic.Deck()  # Note the parentheses here
    deck.shuffle()  # Shuffle the deck before starting the loop

    while True:
        print("\nActions:")
        print("1. Draw card")
        print("2. Check remaining cards")
        print("3. Exit")
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
            break  # Exit the loop
        else:
            print("Invalid input. Please enter a number between 1 and 3.")

if __name__ == "__main__":
    main()