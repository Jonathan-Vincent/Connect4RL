# Connect4RL
Implementing Reinforcement Learning agents to play Connect 4

We recommend to first run train.py to improve the algorithm and then run demo.py and play_against_ai.py.

Connect_4.py contains the game
MCTS.py contains our implementation of the MCTS with UCB. One specialty we are using is, that we use a dictionary with the game states and results that can be saved and loaded for future playthroughs

train_Q.py contains a file that trains and saves Q
demo.py uses the pretrained Q to simulate and output one game
play_against_ai.py is a file where you can play against our algorithm

optimize_c.py is a very basic script to find a good value for c

Q.json is the saved dict
