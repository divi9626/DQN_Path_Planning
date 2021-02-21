## Building environment
import numpy as np
import pickle
import cv2
import sys

import imageio

action_space = 4
Size = 11
HEIGHT = 11
WIDTH = 11
BLUE_CLR = (255,0,0)
WHITE_CLR = (255,255,255)

q_table = np.random.randint(-5,0, (11,11,4))


def obstacle_here(i,j):
    
    
    if (i-5)**2 + (j-5)**2 <= 1:
        return True 
#    
    if (i-3)**2 + (j-8)**2 <= 1:
        return True
#    
#    if (i-7)**2 + (j-8)**2 <= 1:
#        return True
    pass

# reward values
if obstacle_here(9,9):
    print("Goal is within the obstacle")
    sys.exit()
else:
    print("Goal is not within the obstacle")
    
obstacle_penalty = 2000
goal_reward = 1000
step_penalty = 1
wall_penalty = 2000

#Hyperparameters.
Learning_rate = 0.1
Episodes = 10000
Discount = 0.99
epsilon = 0.9
decay_start = 1
decay_end = Episodes//2
decay = epsilon/(decay_end - decay_start)



initial_state = (2,2)
state = initial_state

#epsilon


# move functions will take arguments from the current state tuple
#class Agent:
#    def __init__(self, start):
#        self.x = start[0]
#        self.y = start[1]

def moveup(i, j):
    j = j - 1
    if j <= 0:
        j = 0
    return (i,j)

def  movedown(i,j):
    j = j + 1
    if j >= 10:
        j = 10
    return (i,j)

def moveright(i, j):
    i = i + 1
    if i >= 10:
        i = 10
    return (i,j)

def moveleft(i,j):
    i = i - 1
    if i <= 0:
        i = 0
    return (i,j)

#initial state will passed to the action function and will be updated (return move will be new state)    
    
def newState(state, action_choice):
    i = state[0]
    j = state[1]    

    if action_choice == 0:
        move = moveup(i,j)
        return move
    if action_choice == 1:
        move = movedown(i,j)
        return move
    if action_choice == 2:
        move = moveright(i,j)
        return move
    if action_choice == 3:
        move = moveleft(i,j)
        return move
    

def rewards(state):
    goal_state = (9,9)
    reward = 0
    if obstacle_here(state[0], state[1]):
        reward = -obstacle_penalty
    elif state == goal_state:
        reward = goal_reward
    else:
        reward = -step_penalty
    return reward
        
counter = 0
for episode in range(Episodes):
    done = False
    state = initial_state
    while not done:
        counter = counter + 1
        #print("I'm inside")
        if np.random.random() > epsilon:
            action = np.argmax(q_table[state])
        else:
            action = np.random.randint(0,4)
        
        current_q = q_table[state + (action,)]
        
        new_state = newState(state, action)
        reward = rewards(new_state)
              
        future_q = np.max(q_table[new_state])
            
    
        new_q_value = (1 - Learning_rate) * current_q + Learning_rate * (reward + Discount * future_q)
        
        if reward == -obstacle_penalty:
            new_q_value = -obstacle_penalty
            q_table[state + (action,)] = new_q_value
            print("obstacle")
            done = True
        if reward == goal_reward:
            new_q_value = goal_reward
            q_table[state + (action,)] = new_q_value
            print("Hurray! I made it")
            done  = True
            
        q_table[state + (action,)] = new_q_value
        
        state = new_state 
    if decay_end >= episode >= decay_start:
        epsilon = epsilon - decay
        
    


def get_config_space():
    config_space = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    for i in range(HEIGHT):
        for j in range(WIDTH):
            if (obstacle_here(i, j)):
                config_space[i, j] = BLUE_CLR
    return config_space






def visualize_path(visited_states):
    config_space = get_config_space()
    total = len(visited_states)
    images = []
    for ind, state in enumerate(visited_states):
        config_space[state] = WHITE_CLR
        
        
        if ind == 0:
            config_space[state] = (0,0,255)
        if ind == total-1:
            config_space[state] = (0,255,255)
            
            
        resized = cv2.resize(config_space, (500, 500), interpolation = cv2.INTER_AREA)
        
        filename = f'output_0000{ind}.png'
        cv2.imwrite(filename, resized)
        images.append(imageio.imread(filename))
        
        
        cv2.imshow('Environment Map', resized)
        if cv2.waitKey(100) == 27:
            break
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    imageio.mimsave("output.gif", images)
    
    
    
    
    
    
    
    
    
def test(q_table):
    start_node = (1,2)
    state = start_node
    goal_node = (9,9)
    i = 0
    MAX_ITER = 10000
    visited_states = []
    while state != goal_node and i < MAX_ITER:
        visited_states.append(state)
        action = np.argmax(q_table[state])
        new_state = newState(state, action)
        state = new_state
        i += 1
    if i < MAX_ITER:
        print("Reached")
    else:
        print("too bad")
    
    return visited_states


print("Going to test\n")    
visited_states = test(q_table)

print("Starting visualization...")
visualize_path(visited_states)

        
