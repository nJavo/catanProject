import boardLogic
import devLogic

def main():
    board = boardLogic.create_board()

    boardLogic.print_board(board)

    deck = devLogic.Deck()
    deck.initialize_deck()
    deck.shuffle()
    print(deck.draw_card().type)

if __name__ == "__main__":
    main()