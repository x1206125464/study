import numpy as np
import random

# create Q matrix and R matrix
Q_matrix = np.zeros([6,6])
Q_matrix = np.matrix(Q_matrix)

R_matrix = np.array([[-1,  -1,  -1,  -1,   0,  -1],
                     [-1,  -1,  -1,   0,  -1, 100],
                     [-1,  -1,  -1,   0,  -1,  -1],
                     [-1,   0,   0,  -1,   0,  -1],
                     [ 0,  -1,  -1,   0,  -1, 100],
                     [-1,   0,  -1,  -1,   0, 100]])
R_matrix = np.matrix(R_matrix)

# set gamma value
gamma = 0.8

# training Q matrix
for i in range(500):
    # get beginning state
    begin_state = random.randint(0, 5)
    state = begin_state
    # training main loop
    while(1):
        optional_action = []
        for action in range(6):
            if R_matrix[state, action] >= 0:
                optional_action.append(action)
        action_chose = optional_action[random.randint(0,len(optional_action)-1)]

        optional_Q = []
        for action in range(6):
            if R_matrix[action_chose, action] >= 0:
                optional_Q.append(Q_matrix[action_chose, action])

        Q_matrix[state, action_chose] = R_matrix[state, action_chose] + gamma * np.array(optional_Q).max()
        state = action_chose
        if state == 5:
            break

# get Q martix after training 
Q_matrix = np.uint(Q_matrix/np.array(Q_matrix).max()*100)
print('Q_matrix \n',Q_matrix)

# test episode
for i in range(10):
    begin_state = random.randint(0, 5)
    print('agent born in :',begin_state)
    state = begin_state
    while(1):
        MAX = np.array(Q_matrix[state]).max()
        optional_action = []
        for action in range(6):
            if Q_matrix[state, action] == MAX:
                optional_action.append(action)
                
        action_chose = optional_action[random.randint(0,len(optional_action)-1)]
        print('agent go to :',action_chose)

        state = action_chose
        if state == 5:
            print('agent go out \n')
            break

