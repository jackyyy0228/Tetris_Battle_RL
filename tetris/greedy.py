# test by greedy algorithm
import numpy as np
from tetris import Tetris,Screen
import time 
import scipy as sp
import scipy.stats

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    return m, m-h, m+h

n_test = 100
T = Tetris(action_type = 'grouped',use_fitness = False,is_display = False)
n_lines = []
n_cleared = []
n_length = []
for i in range(n_test):
    state = T.reset()
    done = False
    total_line = 0
    total_length = 0
    total_cleared = 0
    while not done:
        #q_array = T.all_possible_move()
        #actionID = np.argmax(q_array)
        actionID = np.random.randint(53)
        state,_,done,line_sent,line_cleared = T.step(actionID)
        total_line += line_sent
        total_length += 1
        total_cleared += line_cleared
        if total_length >= 200:
            break
    n_lines.append(total_line)
    n_length.append(total_length)
    n_cleared.append(total_cleared)
    print("Test " + str(i) + ": Total line cleared : " + str(total_cleared) + " Total lines sent : " + str(total_line) + " Total length:" + str(total_length))
mean_lines = np.mean(n_lines)
mean_length = np.mean(n_length)
mean_cleared = np.mean(n_cleared)
a,m,b=mean_confidence_interval(n_cleared)
print("Variance of line cleared :" + str(np.var(n_cleared)) )
print("Mean of line cleared : " + str(a) + ' ' + str(mean_cleared) + ' ' + str(b))
print("Mean of line sent : " + str(mean_lines))
print("Mean of game length : " + str(mean_length))
        

    

