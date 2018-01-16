# Author: Georgios Kokkinos
# Implementing http://mnemstudio.org/path-finding-q-learning-tutorial.htm
# Based on https://gist.github.com/kastnerkyle/d127197dcfdd8fb888c2 from Kyle Kastner
# Q-learning formula from http://sarvagyavaish.github.io/FlappyBirdRL/


import numpy as np
from collections import Counter
import json


def update_q(state, next_state, action, alpha, gamma):
    rsa = r[state, action]
    qsa = q[state, action]
    new_q = qsa + alpha * (rsa + gamma * max(q[next_state, :]) - qsa)
    q[state, action] = new_q
    # renormalize row to be between 0 and 1
    rn = q[state][q[state] > 0] / np.sum(q[state][q[state] > 0])
    q[state][q[state] > 0] = rn
    return r[state, action]

#---ideally do this on the engine---
def show_traverse(endState):
    #list with states and number choosen
    traversals = []
    #show all the greedy traversals
    for i in range(len(q)):
        #current_state = i
        #traverse = "%i" % current_state +","
        current_state = i
        traverse = statesNames[i] +","
        n_steps = 0
        #change hardcoding state thingy
        while current_state != endState and n_steps < 20:
            next_state = np.argmax(q[current_state])
            current_state = next_state
            #traverse += "%i -> " % current_state
            traverse += statesNames[current_state] +","
            n_steps = n_steps + 1
        # cut off final arrow
        traverse = traverse[:-1]
        print("Greedy traversal for starting state " +statesNames[i])
        print(traverse)
        print("")
        traversals.append(traverse)
    return traversals

#--------------------------------------------------initial import stuff------------------------------------------------------------
lines = [line.rstrip('\n') for line in open('MyFileName.sav')]

tempStatesAndActions = []
statesAndActions = []
statesNames = []

#seperate stateAndActions from state names
for i in range(1,len(lines),2):
     tempStatesAndActions.append(lines[i])

for i in range(0,len(lines),2):
    statesNames.append(lines[i])

#convert to int and np array
temp = []
for i in range(len(tempStatesAndActions)):
    temp =[x.strip() for x in tempStatesAndActions[i].split(',')]
    temp.remove("")
    temp =[int(x) for x in temp]
    statesAndActions.append(temp)

r = np.array(statesAndActions).astype("float32")
print(r)
print(statesNames)
q = np.zeros_like(r)

#get end state to stop traversal
endStateList = max(statesAndActions)
endState = max(endStateList)
print('endstate is ' + str(endStateList.index(endState)))
endStateNum = endStateList.index(endState)
# -----------------------------------------------Core algorithm------------------------------------------------------------
gamma = 0.8
alpha = 1.
n_episodes = 1E3

#n_states = 5
#n_actions = 5
n_states = len(statesNames)
n_actions = len(statesNames)
epsilon = 0.05
random_state = np.random.RandomState(1999)
for e in range(int(n_episodes)):
    states = list(range(n_states))
    random_state.shuffle(states)
    current_state = states[0]
    goal = False
    if e % int(n_episodes / 10.) == 0 and e > 0:
        pass
        # uncomment this to see plots each monitoring
        # show_traverse()
    while not goal:
        # epsilon greedy
        valid_moves = r[current_state] >= 0
        if random_state.rand() < epsilon:
            actions = np.array(list(range(n_actions)))
            actions = actions[valid_moves == True]
            if type(actions) is int:
                actions = [actions]
            random_state.shuffle(actions)
            action = actions[0]
            next_state = action
        else:
            if np.sum(q[current_state]) > 0:
                action = np.argmax(q[current_state])
            else:
                # Don't allow invalid moves at the start
                # Just take a random move
                actions = np.array(list(range(n_actions)))
                actions = actions[valid_moves == True]
                random_state.shuffle(actions)
                action = actions[0]
            next_state = action
        reward = update_q(current_state, next_state, action,
                          alpha=alpha, gamma=gamma)
        # Goal state has reward 
        if reward > 21:
            goal = True
        current_state = next_state

print(q)
traversals = show_traverse(endStateNum)
traversals2 =[]
for i in range(len(traversals)):
    temp =[x.strip() for x in traversals[i].split(',')]
    traversals2.append(temp)

#put them in one list and count them
flat_list = [item for sublist in traversals2 for item in sublist]
nn = Counter(flat_list)

#print(flat_list)
print(traversals2[0])
print(nn)
print(nn['BP-TestShout2'])

#---write to file---
open('C:/Users/lalai/Documents/GitHub/Thesis/qResults.json', 'w').close()
file = open('C:/Users/lalai/Documents/GitHub/Thesis/qResults.json','w') 

data = []
for i in range(len(statesNames)):
    data.append({
    'Name' : str(i),
    'testShoutName' : statesNames[i],
    'timesAppeared' : nn[statesNames[i]]
    })
json.dump(data,file)
file.close()

    
#show_q()