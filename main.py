from Connect_4 import Connect_four
from MCTS_2 import MCTS
import time, os

def run(game):
    res = None
    while res == None:
        mcts = MCTS(game, n_branches=100, c=2, symmetry=False)
        chosen_move = mcts.run()
        #catch draws
        if chosen_move in ["draw", "player1", "player2"]:
            return chosen_move
        
        game.add_move(chosen_move)
        _ = os.system('cls')
        game.print_state()
        #time.sleep(0.5)

def main():
    game = Connect_four()
    run(game)

if __name__ == "__main__":
    main()