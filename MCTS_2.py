import numpy as np
from collections import defaultdict
from Connect_4 import Connect_four

class MCTS:
    def __init__(self, game: Connect_four, n_branches=100, c=2, symmetry=True):
        self.game = game
        self.n_branches = n_branches
        self.c = c
        self.symmetry = symmetry
        self.base_move = self.game.moves_played
        # Q has values of form [draws, player1, player2]
        self.Q = defaultdict(lambda: [0,0,0])
        
    def policy(self, explore = 1):
        actions = self.game.legal_moves()
        if len(actions) == 1:
            return actions[0]
        
        N = np.sum(self.Q[self.game.state_to_string()])
        
        if N == 0:
            return np.random.choice(actions)
        
        best = -np.inf
        for action in actions:
            self.game.add_move(action)
            results = self.Q[self.game.state_to_string()]
            n = sum(results)+0.001 #for stability
            UCT = (results[self.game.turn * -1] - results[self.game.turn])/n + explore*self.c*np.sqrt(np.log(N)/n)
            if UCT > best:
                best = UCT
                best_action = action
            self.game.remove_move()
        return best_action
    
    
    def rollout_policy(self):
        actions = self.game.legal_moves()
        return np.random.choice(actions)
    
    def make_move(self):
        action = self.rollout_policy()
        self.game.add_move(action)
    
    def rollout(self):
        while self.game.result() == None:
            self.make_move()
        return self.game.result()
    
    def backprop(self, result=None):
        
        string_state = self.game.state_to_string()
        self.Q[string_state][result] += 1
        
        if self.symmetry:
            sym_state = self.game.state.T[::-1].T
            string_sym_state = self.game.state_to_string(sym_state)
            self.Q[string_sym_state][result] += 1
        
        if np.sum(abs(self.game.state)) == self.base_move:
            return
        self.game.remove_move()
        self.backprop(result)
    
    def run(self):
        if self.game.result() != None:
            return ["draw", "player1", "player2"][self.game.result()]
                
        #run branches MCTS runs
        for _ in range(self.n_branches):
            result = self.rollout()
            self.backprop(result)
            
        return(self.policy())