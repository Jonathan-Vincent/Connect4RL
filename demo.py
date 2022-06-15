from Connect_4 import Connect_four
from MCTS import MCTS
import json
import time, os

def load_Q(filename):
    with open(filename, 'r') as f:
        Q = json.load(f)
    return Q

def run(game):
    Q = load_Q('Q.json')
    while True:
        mcts = MCTS(game, n_branches=100, c=10000, symmetry=True)
        mcts.Q.update(Q)
        chosen_move = mcts.run()
        Q = mcts.Q
        
        if chosen_move in ["draw", "player1", "player2"]:
            return chosen_move
        
        game.add_move(chosen_move)
        _ = os.system('cls')
        game.print_state()
        time.sleep(0.5)

def main():
    game = Connect_four()
    print(run(game))

if __name__ == "__main__":
    main()