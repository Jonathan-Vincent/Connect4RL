# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm
from collections import defaultdict

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
        global diag_dict
        self.nrows = nrows
        self.ncols = ncols
        self.base_state = np.zeros((nrows,ncols))
        self.turn = 1
        self.Q = defaultdict(lambda: 0)
        self.starting_moves = [i for i in range(ncols)]
        self.last_move = (0,0)
        self.visited = []

    #checks if the last move results in a win
    def is_win(self,state,move):
        
        diags = diag_dict[move]
        #iterate through
        for diag in diags:
            if abs(np.sum(state[diag])) == 4:
                game.outcome = ("win",self.turn,diag)
                return self.turn
            
        return 0
    
    def legal_moves(self,state):
        allowed = (np.sum(abs(state),axis=0)< self.nrows)
        return np.where(allowed)[0]
            
                
    # add move to current board state
    def add_move(self,state,move):
        
        for row in range(self.nrows):
            if state[row,move] == 0:
                state[row,move] = self.turn
                return state,(row,move)
    
    def depth_search(self,state):
        
        string_state = ''.join([str(int(i+2)) for i in state.flatten()])
        if string_state not in self.visited:
            self.visited.append(string_state)
        
        actions = self.legal_moves(state)
        if len(actions) == 0:
            return 0
        
        a = np.random.choice(actions)
        state_p, self.last_move = self.add_move(state,a)
        self.turn *= -1
        
        v = self.is_win(state_p,self.last_move)
        if v != 0:
            return v
        
        v = self.depth_search(state_p)
        
        return v
    

diag_dict = generate_diag_dict()
game = connect_four()

game.depth_search(game.base_state)