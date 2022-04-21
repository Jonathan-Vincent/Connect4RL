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
        self.turn = "p1"
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
        if self.turn == "p1":
            board = self.state_p1
        else:
            board = self.state_p2
            
        #print(board)
        
        row,col = self.last_move
        diags = self.diag_dict[(row,col)]
        #iterate through
        for diag in diags:
            if np.sum(board[diag]) == 4:
                game.outcome = ("win",self.turn,diag)
                return True
            
        return False
                
    # add move to current board state
    def add_move(self,move):
        self.move_count[move] += 1
        if self.move_count[move] == self.nrows:
            self.legal_moves.remove(move)
        for row in range(self.nrows):
            if self.state[row,move] == 0:
                if self.turn == "p1":
                    self.state_p1[row,move] = 1
                    self.state[row,move] = 1
                else:
                    self.state_p2[row,move] = 1
                    self.state[row,move] = -1
                
                return row,move
    
    #play a game between agent1 and agent2
    def run(self,agent1,agent2,verbose=False):
        
        for i in range(self.nrows*self.ncols):
            if i % 2 == 0:
                game.turn = "p1"
                move = agent1(self.state,self.legal_moves, 1)
            else:
                game.turn = "p2"
                move = agent2(self.state,self.legal_moves, -1)
            
            self.last_move = self.add_move(move)
            if verbose:
                print(self.state,i)
            if self.is_win():
                return
        
        self.outcome = ("Tie","")
            

class Agent():
    def __init__(self,move_func,learn=False):
        self.get_move = move_func
        self.learn = learn
    
    def __call__(self, state, legal_moves, obj):
        if len(legal_moves) == 1:
            return legal_moves[0]
        return self.get_move(state, legal_moves, obj)
            

def random_move(state, legal_moves, obj):
    return np.random.choice(legal_moves)


agent1 = Agent(random_move)
agent2 = Agent(random_move)

p1_wins = 0
p2_wins = 0
for i in tqdm(range(1000)):
    game = connect_four()
    game.run(agent1,agent2,verbose=False)
    #print(i,game.outcome)
    if game.outcome[1] == "p1":
        p1_wins += 1
    elif game.outcome[1] == "p2":
        p2_wins += 1

print(p1_wins,p2_wins)
