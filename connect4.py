# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm

class connect_four():
    def __init__(self,nrows=6,ncols=7):
        self.state_p1 = np.zeros((nrows,ncols))
        self.state_p2 = np.zeros((nrows,ncols))
        self.state = np.zeros((nrows,ncols))
        self.nrows = nrows
        self.ncols = ncols
        self.turn = 1
        self.outcome = None
        self.move_count = [0 for i in range(ncols)]
        self.legal_moves = [i for i in range(ncols)]
        self.diag_dict = {}
        self.generate_diag_dict()
    
    #diag_dict is a dictionary storing all of the possible 4 in a row combinations
    #each (i,j) position is a key
    #the values are the list of 4-vectors that go through (i,j)
    #so diag_dict[(0,0)] returns
    #[((0, 1, 2, 3), (0, 0, 0, 0)),
    # ((0, 0, 0, 0), (0, 1, 2, 3)),
    # ((0, 1, 2, 3), (0, 1, 2, 3))]
    def generate_diag_dict(self):
        #rows
        for row in range(self.nrows-3):
            for col in range(self.ncols):
                row_tup = [row + i for i in range(4)]
                col_tup = [col for i in range(4)]
                for pos in zip(row_tup,col_tup):
                    if pos not in self.diag_dict.keys():
                        self.diag_dict[pos] = [(tuple(row_tup),tuple(col_tup))]
                    else:
                        self.diag_dict[pos] = self.diag_dict[pos]  + [(tuple(row_tup),tuple(col_tup))]
                        
        #cols
        for row in range(self.nrows):
            for col in range(self.ncols-3):
                row_tup = [row for i in range(4)]
                col_tup = [col+i for i in range(4)]
                for pos in zip(row_tup,col_tup):
                    if pos not in self.diag_dict.keys():
                        self.diag_dict[pos] = [(tuple(row_tup),tuple(col_tup))]
                    else:
                        self.diag_dict[pos] = self.diag_dict[pos]  + [(tuple(row_tup),tuple(col_tup))]
                        
        
        #right diagonals
        for row in range(self.nrows-3):
            for col in range(self.ncols-3):
                row_tup = [row+i for i in range(4)]
                col_tup = [col+i for i in range(4)]
                for pos in zip(row_tup,col_tup):
                    if pos not in self.diag_dict.keys():
                        self.diag_dict[pos] = [(tuple(row_tup),tuple(col_tup))]
                    else:
                        self.diag_dict[pos] = self.diag_dict[pos]  + [(tuple(row_tup),tuple(col_tup))]
                        
        #left diagonals
        for row in range(3,self.nrows):
            for col in range(self.ncols-3):
                row_tup = [row-i for i in range(4)]
                col_tup = [col+i for i in range(4)]
                for pos in zip(row_tup,col_tup):
                    if pos not in self.diag_dict.keys():
                        self.diag_dict[pos] = [(tuple(row_tup),tuple(col_tup))]
                    else:
                        self.diag_dict[pos] = self.diag_dict[pos]  + [(tuple(row_tup),tuple(col_tup))]
        
        
    #checks if the last move results in a win
    def is_win(self):
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
            
        return False
                
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
                self.turn *= -1
                self.last_move = row,move
                break
    
    #play a game between agent1 and agent2
    def run(self,agent1,agent2,verbose=False):
        
        for i in range(self.nrows*self.ncols) - sum(self.move_count):
            if self.turn == 1:
                move = agent1(self,self.legal_moves, 1)
            else:
                move = agent2(self,self.legal_moves, -1)
            
            self.add_move(move)
            if verbose:
                print(self.state,i)
            if self.is_win():
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
        if game_copy.is_win():
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
                if game_copy_2.is_win():
                    continue
                game_copy_2.add_move(multi_step_lookahead(game_copy_2, game_copy_2.legal_moves, -obj, k-1))
                if game_copy_2.is_win():
                    return i
            game_copy = deepcopy(game)
    return np.random.choice(legal_moves)


def run_simulations(agent_1_strategy, agent_2_strategy, n_games, verbose=False):
    agent1 = Agent(agent_1_strategy)
    agent2 = Agent(agent_2_strategy)

    p1_wins = 0
    p2_wins = 0
    game = connect_four()
    for i in tqdm(range(n_games)):
        game.run(agent1,agent2,verbose=verbose)
        #print(i,game.outcome)
        if game.outcome[1] == "p1":
            p1_wins += 1
        elif game.outcome[1] == "p2":
            p2_wins += 1
        game.reset()

    print(p1_wins,p2_wins)
    
def main():
    run_simulations()

if __name__ == "__main__":
    main()
