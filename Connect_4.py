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


class Connect_four():
    def __init__(self,nrows=6,ncols=7):
        self.diag_dict = generate_diag_dict(nrows,ncols)
        self.nrows = nrows
        self.ncols = ncols
        self.max_moves = nrows*ncols
        self.state = np.zeros((nrows,ncols))
        self.turn = 1
        self.moves_played = 0
        self.starting_moves = [i for i in range(ncols)]
        self.history = []

    #checks if the last move results in a win
    def result(self):
        if self.moves_played == self.max_moves:
            return 0
        
        if self.moves_played == 0:
            return None
        
        else:
            diags = self.diag_dict[self.history[-1]]
            #iterate through
            for diag in diags:
                if np.sum(self.state[diag]) == 4:
                    return 1
                elif np.sum(self.state[diag]) == -4:
                    return -1
                
            return None
    
    def legal_moves(self):
        allowed = (np.sum(abs(self.state),axis=0) < self.nrows)
        return np.where(allowed)[0]
            
    # add move to current board state
    def add_move(self, column: int):
        row = np.argwhere(self.state[:,column] == 0)[0][0]
        self.state[row,column] = self.turn
        self.turn *= -1
        self.moves_played += 1
        self.history.append((row,column))
            
    def remove_move(self):
        self.state[self.history.pop()] = 0
        self.turn *= -1
        self.moves_played -= 1
    
    def state_to_string(self, state = None):
        if type(state) == np.ndarray:
            return ''.join(['X' if x == 1 else 'O' if x == -1 else '.' for x in state.flatten()])
        return ''.join(['X' if x == 1 else 'O' if x == -1 else '.' for x in self.state.flatten()])
    
    def string_to_state(self, string):
        return np.array([[1 if x == 'X' else -1 if x == 'O' else 0 for x in row] for row in string.split('\n')]).reshape(self.nrows,self.ncols)
    
    def print_state(self):
        print("\n".join([''.join(['X' if x == 1 else 'O' if x == -1 else '.' for x in row]) for row in self.state[::-1, :]]))
        