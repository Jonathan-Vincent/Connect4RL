# -*- coding: utf-8 -*-
import numpy as np
from tqdm import tqdm
from copy import deepcopy



#diag_dict is a dictionary storing all of the possible 4 in a row combinations
#each (i,j) position is a key
#the values are the list of 4-vectors that go through (i,j)
#so diag_dict[(0,0)] returns
#[((0, 1, 2, 3), (0, 0, 0, 0)),
# ((0, 0, 0, 0), (0, 1, 2, 3)),
# ((0, 1, 2, 3), (0, 1, 2, 3))]
    
def generate_diag_dict(nrows=6,ncols=7):
    diag_dict = {}
    #rows
    for row in range(nrows-3):
        for col in range(ncols):
            row_tup = [row + i for i in range(4)]
            col_tup = [col for i in range(4)]
            for pos in zip(row_tup,col_tup):
                if pos not in diag_dict.keys():
                    diag_dict[pos] = [(tuple(row_tup),tuple(col_tup))]
                else:
                    diag_dict[pos] = diag_dict[pos]  + [(tuple(row_tup),tuple(col_tup))]
                    
    #cols
    for row in range(nrows):
        for col in range(ncols-3):
            row_tup = [row for i in range(4)]
            col_tup = [col+i for i in range(4)]
            for pos in zip(row_tup,col_tup):
                if pos not in diag_dict.keys():
                    diag_dict[pos] = [(tuple(row_tup),tuple(col_tup))]
                else:
                    diag_dict[pos] = diag_dict[pos]  + [(tuple(row_tup),tuple(col_tup))]
                    
    
    #right diagonals
    for row in range(nrows-3):
        for col in range(ncols-3):
            row_tup = [row+i for i in range(4)]
            col_tup = [col+i for i in range(4)]
            for pos in zip(row_tup,col_tup):
                if pos not in diag_dict.keys():
                    diag_dict[pos] = [(tuple(row_tup),tuple(col_tup))]
                else:
                    diag_dict[pos] = diag_dict[pos]  + [(tuple(row_tup),tuple(col_tup))]
                    
    #left diagonals
    for row in range(3,nrows):
        for col in range(ncols-3):
            row_tup = [row-i for i in range(4)]
            col_tup = [col+i for i in range(4)]
            for pos in zip(row_tup,col_tup):
                if pos not in diag_dict.keys():
                    diag_dict[pos] = [(tuple(row_tup),tuple(col_tup))]
                else:
                    diag_dict[pos] = diag_dict[pos]  + [(tuple(row_tup),tuple(col_tup))]
                    
    return diag_dict

class connect_four():
    def __init__(self,nrows=6,ncols=7):
        self.state_p1 = np.zeros((nrows,ncols))
        self.state_p2 = np.zeros((nrows,ncols))
        self.state = np.zeros((nrows,ncols))
        self.nrows = nrows
        self.ncols = ncols
        self.turn = 1
        self.outcome = None
        self.legal_moves = [i for i in range(ncols)]
        self.move_count = [0 for i in range(ncols)]
        self.diag_dict = generate_diag_dict(nrows,ncols)

    #checks if the last move results in a win
    def check_win(self):
        if self.turn == 1:
            board = self.state_p1
        else:
            board = self.state_p2
            
        #print(board)
        
        row,col = self.last_move
        diags = self.diag_dict[(row,col)]
        #iterate through
        for diag in diags:
            if np.sum(board[diag]) == 4:
                self.outcome = ("win",self.turn,diag)
                if self.turn == 1:
                    return(1)
                if self.turn == -1:
                    return(-1)

                
    # add move to current board state
    def add_move(self,move):
        self.move_count[move] += 1
        if self.move_count[move] == self.nrows:
            self.legal_moves.remove(move)
        for row in range(self.nrows):
            if self.state[row,move] == 0:
                if self.turn == 1:
                    self.state_p1[row,move] = 1
                    self.state[row,move] = 1
                else:
                    self.state_p2[row,move] = 1
                    self.state[row,move] = -1
                self.last_move = row,move
                self.check_win()
                self.turn *= -1
                break
    
    #play a game between agent1 and agent2
    def run(self,agent1,agent2,verbose=False):
        
        for i in range(self.nrows*self.ncols - int(np.sum(self.move_count))) :
            if self.turn == 1:
                move = agent1(self,self.legal_moves, 1)
            else:
                move = agent2(self,self.legal_moves, -1)
            
            self.add_move(move)
            if verbose:
                print(self.state,i)
            if self.outcome:
                return
        
        self.outcome = ("Tie","")
        

            
    def change_state(self, state):
        self.state = state
        self.state_p1 = self.state[self.state == 1]
        self.state_p2 = self.state[self.state == -1]
        self.nrows = self.state.shape[0]
        self.ncols = self.state.shape[1]
        self.turn = 2*np.sum(self.state) - 1
        self.outcome = None
        self.move_count = sum(np.abs(self.state), axis = 1)
        self.legal_moves = [i for i in range(self.ncols)]
        for i in range(self.ncols):
            if self.move_count[i] >= self.nrows: 
                self.legal_moves.remove(i)
        
    def reset(self):
        self.state = np.zeros((self.nrows,self.ncols))
        self.state_p1 = np.zeros((self.nrows,self.ncols))
        self.state_p2 = np.zeros((self.nrows,self.ncols))
        self.turn = 1
        self.outcome = None
        self.move_count = np.zeros(self.ncols)
        self.legal_moves = [i for i in range(self.ncols)]

class Agent():
    def __init__(self,move_func,learn=False):
        self.get_move = move_func
        self.learn = learn
    
    def __call__(self, state, legal_moves, obj):
        if len(legal_moves) == 1:
            return legal_moves[0]
        return self.get_move(state, legal_moves, obj)
            

def random_move(game, legal_moves, obj):
    return np.random.choice(legal_moves)

def one_step_lookahead(game, legal_moves, obj):
    game_copy = deepcopy(game)
    for move in legal_moves:
        game_copy.add_move(move)
        game_copy.check_win()
        if game_copy.outcome:
            return move
        game_copy = deepcopy(game)
    return np.random.choice(legal_moves)

def multi_step_lookahead(game, legal_moves, obj, k = 2):
    game_copy = deepcopy(game)
    if k == 1:
        return(one_step_lookahead(game_copy, game_copy.legal_moves, obj))
    else:
        for i in legal_moves:
            game_copy.add_move(i)
            for j in game.legal_moves:
                game_copy_2 = deepcopy(game_copy)
                game_copy_2.add_move(j)
                game_copy_2.check_win()
                if game_copy_2.outcome:
                    continue
                game_copy_2.add_move(multi_step_lookahead(game_copy_2, game_copy_2.legal_moves, -obj, k-1))
                game_copy_2.check_win()
                if game_copy_2.outcome:
                    return i
            game_copy = deepcopy(game)
    return np.random.choice(legal_moves)

def middle(game, legal_moves, obj):
    return legal_moves[len(legal_moves)//2]

def run_simulations(agent_1_strategy, agent_2_strategy, n_games, verbose=False):
    agent1 = Agent(agent_1_strategy)
    agent2 = Agent(agent_2_strategy)

    p1_wins = 0
    p2_wins = 0
    game = connect_four()
    for _ in tqdm(range(n_games)):
        game.run(agent1,agent2,verbose=verbose)
        #print(i,game.outcome)
        if game.outcome[1] == 1:
            p1_wins += 1
        elif game.outcome[1] == -1:
            p2_wins += 1
        game.reset()
    print(p1_wins,p2_wins)
    
def main():
    agent_1_strategy = one_step_lookahead
    agent_2_strategy = random_move
    n_games = 1000
    verbose=False
    run_simulations(agent_1_strategy, agent_2_strategy, n_games, verbose=verbose)

if __name__ == "__main__":
    main()
