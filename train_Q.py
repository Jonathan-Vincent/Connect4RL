from Connect_4 import Connect_four
from MCTS import MCTS
import json
from tqdm import tqdm


def load_Q(filename):
    with open(filename, 'r') as f:
        Q = json.load(f)
    return Q

def save_Q(Q, filename):
    Q = {key: value for key, value in Q.items() if sum(value) > 1}
    with open(filename, 'w') as f:
        json.dump(Q, f)

def run(game):
    Q = load_Q('Q.json')
    while True:
        mcts = MCTS(game, n_branches=1000, c=10000, symmetry=True)
        mcts.Q.update(Q)
        chosen_move = mcts.run()
        Q = mcts.Q
        if chosen_move in ["draw", "player1", "player2"]:
            # only save states that have been visited at least 10 times to save memory
            Q = {key: value for key, value in Q.items() if sum(value) > 10}
            save_Q(Q, 'Q.json')
            return chosen_move
        game.add_move(chosen_move)

def main():
    for _ in tqdm(range(2)):
        game = Connect_four()
        run(game)

if __name__ == "__main__":
    main()