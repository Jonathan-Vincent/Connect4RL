from Connect_4 import Connect_four
from MCTS import MCTS
import numpy as np

def mcts_move(game, c):
    mcts = MCTS(game, n_branches=10, c=c, symmetry=True)
    chosen_move = mcts.run()
    if chosen_move in ["draw", "player1", "player2"]:
        return chosen_move
    
    game.add_move(chosen_move)
    return(None)

def compete(c_1, c_2):
    result = None
    game = Connect_four()
    while result is None:
        result = mcts_move(game, c_1)
        result = mcts_move(game, c_2)
    if result == "player1":
        return c_1
    elif result == "player2":
        return c_2
    else: 
        return (c_1 + c_2) / 2
    
def optimize_c(n_steps):
    # 10000 has worked well in the past so we use it as a starting value
    c = 10000
    for i in range(n_steps):
        d = np.random.normal(0, c/(10+i))
        c = compete(c+d, c-d)
        print(c, d)
    return(c)
        

def main():
    n_steps = 100
    print("Best c: ", optimize_c(n_steps))

if __name__ == "__main__":
    main()