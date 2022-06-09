from Connect_4 import Connect_four
from MCTS_2 import MCTS
import json
import time, os

def load_Q(filename):
    with open(filename, 'r') as f:
        Q = json.load(f)
    return Q

def save_Q(Q, filename):
    with open(filename, 'w') as f:
        json.dump(Q, f)

def run(game, q_status = ""):
    if "r" in q_status:
        Q = load_Q('Q.json')
    else: 
        Q = {}
    while True:
        mcts = MCTS(game, n_branches=1000, c=10000, symmetry=True)
        mcts.Q.update(Q)
        chosen_move = mcts.run()
        Q = mcts.Q
        
        if chosen_move in ["draw", "player1", "player2"]:
            if "w" in q_status:
                save_Q(Q, 'Q.json')
            return chosen_move
        
        game.add_move(chosen_move)
        _ = os.system('cls')
        game.print_state()
        #time.sleep(0.5)

def main():
    q_status = ""
    results = []
    for _ in range(100):
        game = Connect_four()
        results.append(run(game, q_status))
    print("\n".join(results))

if __name__ == "__main__":
    main()