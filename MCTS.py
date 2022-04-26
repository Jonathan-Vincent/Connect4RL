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


class connect_four_MCTS():
    def __init__(self,nrows=6,ncols=7):
        self.diag_dict = generate_diag_dict(nrows,ncols)
        self.nrows = nrows
        self.ncols = ncols
        self.base_state = np.zeros((nrows,ncols))
        self.turn = 1
        self.Q = defaultdict(lambda: [0,0,0]) #[draws,p1_wins,p2_wins]
        self.starting_moves = [i for i in range(ncols)]
        self.last_move = (0,0)
        self.visited = []
        self.history = []

    #checks if the last move results in a win
    def result(self,state,move):
        if np.sum(abs(state)) == self.nrows*self.ncols:
            return 0
        
        else:
            diags = self.diag_dict[move]
            #iterate through
            for diag in diags:
                if np.sum(state[diag]) == 4:
                    return 1
                elif np.sum(state[diag]) == -4:
                    return -1
                
            return None
    
    def legal_moves(self,state):
        allowed = (np.sum(abs(state),axis=0)< self.nrows)
        return np.where(allowed)[0]
            
                
    # add move to current board state
    def add_move(self,state,column):
        for row in range(self.nrows):
            if state[row,column] == 0:
                state[row,column] = self.turn
                return state,(row,column)
            
    def remove_move(self,state,move):
        state[move[0],move[1]] = 0
        return state
    
    def state_to_string(self,state):
        return ''.join([str(int(i+1)) for i in state.flatten()])
    
    def string_to_state(self,state):
        return np.array([int(x)-1 for x in state]).reshape(self.nrows,self.ncols)
        
    
    def MCTS(self,root,n_branches=10000):
        #node has form (state,parent_move)
        def policy(actions):
            return np.random.choice(actions)
        
        def select(node):
            actions = self.legal_moves(node[0])
            action = policy(actions)
            state_p, move = self.add_move(node[0],action)
            self.history.append(move)
            self.turn *= -1
            return state_p, move
        
        def rollout(node):
            res = self.result(node[0],node[1])
            while res == None:
                node = select(node)
                res = self.result(node[0],node[1])
            return res
        
        def backprop(node,result):
            if np.sum(abs(node[0])) == 0:
                return
            
            string_state = self.state_to_string(node[0])
            self.Q[string_state][result] += 1
            parent = (self.remove_move(node[0],self.history[-1]),self.history.pop())
            backprop(parent,result)
        
        #run branches MCTS runs
        for _ in tqdm(range(n_branches)):
            leaf = select((root,None))  #None indicates there is no parent move
            res = rollout(leaf)
            backprop(leaf,res)
            self.turn = 1
            
    def best_move(self,state,player):
        best_score = 0
        for action in self.legal_moves(state):
            state_p = state.copy()
            state_p, move = self.add_move(state_p, action)
            results = self.Q[self.state_to_string(state_p)]
            
            #score ignores draws
            score = ((results[1] - results[2])*player)/sum(results)
            print(action,results,score)
            if score > best_score:
                best_score = score
                chosen_move = action
                
        return chosen_move, best_score
            
if __name__ == "__main__":
    
    game = connect_four_MCTS()
    
    game.MCTS(game.base_state)
    print("MCTS complete")
    print(game.best_move(game.base_state,1))