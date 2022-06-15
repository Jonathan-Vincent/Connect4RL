from Connect_4 import Connect_four
from MCTS import MCTS
import json
import time, os

def load_Q(filename):
    with open(filename, 'r') as f:
        Q = json.load(f)
    return Q

def save_Q(Q, filename):
    Q = {key: value for key, value in Q.items() if sum(value) > 1}
    with open(filename, 'w') as f:
        json.dump(Q, f)
        
def print_game(game):
    _ = os.system('cls')
    print("1234567")
    game.print_state()
        
def mcts_move(game, Q):
    print_game(game)
    mcts = MCTS(game, n_branches=100, c=10000, symmetry=True)
    mcts.Q.update(Q)
    chosen_move = mcts.run()
    Q = mcts.Q
    if chosen_move in ["draw", "player1", "player2"]:
        return chosen_move
    
    game.add_move(chosen_move)
    return(Q)
    
def player_move(game):
    print_game(game)
    if game.history == []:
        move = int(input("Enter move: "))
    else:
        print("Last move: ", game.history[-1][1] + 1)
        move = int(input("Enter move: "))
    game.add_move(move-1)

def run(game, player):
    Q = load_Q('Q.json')
    if player == 2:
        Q = mcts_move(game, Q)
    while True:
        player_move(game)
        Q = mcts_move(game, Q)
        if Q in ["draw", "player1", "player2"]:
            return(Q)
        

def main():
    player = 1#2
    game = Connect_four()
    print(run(game, player))

if __name__ == "__main__":
    main()