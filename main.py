import boardLogic
import devLogic

def main():
    board = boardLogic.create_board()

    boardLogic.print_board(board)

    deck = devLogic.Deck
    deck.initialize_deck(deck)
    print(deck.draw_card(deck))

if __name__ == "__main__":
    main()