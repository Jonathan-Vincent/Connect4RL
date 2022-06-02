# -*- coding: utf-8 -*-

import numpy as np
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import time as time

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
        self.last_move = (0,0)
        self.history = []

    #checks if the last move results in a win
    def result(self,state,move):
        if np.sum(abs(state)) == self.nrows*self.ncols:
            return 0
        
        else:
            diags = self.diag_dict[move]
            #iterate through
            for diag in diags:
                diag_sum = np.sum(state[diag])
                if diag_sum == 4:
                    return 1
                elif diag_sum == -4:
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
        
    
    def MCTS(self,root,player,n_branches,symmetry=True):
        #node has form (state,parent_move)
        self.history = []
        
        def rollout_policy(state):
            actions = self.legal_moves(state)
            return np.random.choice(actions)
            
        
        def select(node):
            action = rollout_policy(node[0])
            state_p, move = self.add_move(node[0],action)
            self.history.append(move)
            self.turn = -self.turn
            return state_p, move
        
        def rollout(node):
            res = self.result(node[0],node[1])
            while res == None:
                node = select(node)
                res = self.result(node[0],node[1])
            return res
        
        def backprop(node,result):
            if np.sum(abs(node[0])) == moves_played:
                return
            
            string_state = self.state_to_string(node[0])
            self.Q[string_state][result] += 1
            
            if symmetry:
                sym_state = node[0].T[::-1].T
                string_sym_state = self.state_to_string(sym_state)
                self.Q[string_sym_state][result] += 1
            
            parent = (self.remove_move(node[0],self.history[-1]),self.history.pop())
            backprop(parent,result)
        
        
        moves_played = np.sum(abs(root[0]))
        if moves_played >= self.nrows*self.ncols -1:
            return
        #run branches MCTS runs
        for _ in range(n_branches):
            leaf = select(root) 
            res = rollout(leaf)
            backprop(leaf,res)
            self.turn = player
            
    
    def policy(self,state,C=np.sqrt(2)):
        actions = self.legal_moves(state)
        if len(actions) == 1:
            return actions[0]
        
        N = max(2,np.sum(self.Q[self.state_to_string(state)]))
        
        #print("action|exploit|explore|UCT")
        best = -np.inf
        for action in actions:
            state_p = state.copy()
            state_p, move = self.add_move(state_p, action)
            results = self.Q[self.state_to_string(state_p)]
            n = max(1,sum(results))
            exploit = results[self.turn]/n
            explore =  C*np.sqrt(np.log(N)/n)
            UCT = exploit + explore
            #print(action,"    |",np.round(exploit,3),"|",np.round(explore,3),"|",np.round(UCT,3))
            if UCT > best:
                best = UCT
                best_action = action
                
        return best_action
    
    def run_game(self,n_branches,C=np.sqrt(2),use_MCTS=True,verbose=False):
        self.state = self.base_state.copy()
        self.turn = 1
        res = None
        move = (0,0)
        while res == None:
            
            self.MCTS((self.state,move),self.turn,n_branches)
            #print(np.where(self.state ==-1, 2, self.state))
            chosen_move = self.policy(self.state,C=C)
            if not use_MCTS and self.turn == -1:
                allowed = (np.sum(abs(self.state),axis=0)< self.nrows)
                chosen_move = np.random.choice(np.where(allowed)[0])
                    
            #catch draws
            if chosen_move == "draw":
                return 0
            
            self.state, move = self.add_move(self.state, chosen_move)
            res = self.result(self.state,move)
            
            if verbose:
                time.sleep(0.1)
                print(np.where(self.state ==-1, 2, self.state))
            
            self.turn = -self.turn
            
        return res
            
if __name__ == "__main__":
    
    game = connect_four_MCTS()
    
    #plays 100 games
    #average around 20 moves per game, 100 MCTS branches per move
    #~40,000 branches seen
    def anneal(i,n):
        return np.sqrt(2)
        #return 3*(1-(i/n)) + (i/n)
        
    n = 1000
    games = []
    
    for i in range(n):
        game.run_game(64,C=anneal(i,n))
        print(i,"\nC-value:", anneal(i,n), "Game length:",np.sum(abs(game.state)))
        print(np.where(game.state ==-1, 2, game.state))
        games.append(np.sum(abs(game.state)))
        plt.plot(games)
        plt.show()
        
        if i % 10 == 9:
            results = [0,0,0]
            for j in range(10):
                res = game.run_game(64,C=np.sqrt(2),use_MCTS=False,verbose=False)
                if res == -1:
                    print("Win state:",j)
                    print(np.where(game.state ==-1, 2, game.state))
                #print(len(game.Q))
                results[res] += 1
            
            print(i,"Win rate:",results[1]/10,len(game.Q))
    
    
        
   
        
    total_printed = 0
    for k,v in game.Q.items():
        if np.sum(v) > 20:
            total_printed += 1
            print(game.string_to_state(k))
            print(v)
            print(v[1]/sum(v))
            if total_printed == 7:
                break

      