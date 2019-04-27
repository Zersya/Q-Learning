import pandas as pd
import numpy as np
import random

def initRtable(gridWorld):
    rTable = []
    pos = [0,0]
    outArena = -100

    for x in range(len(gridWorld)**2):
        if(x != 0):
            if(x%15 == 0):
                pos[0] += 1
                pos[1] = 0
            else:
                pos[1] += 1  

        arr = []
        y = pos[0]
        x = pos[1]
        for k in range(4):
            if(k == 0): #NORTH
                if(y-1 > -1):
                    arr.append(gridWorld.iloc[y-1,x])
                else:
                    arr.append(outArena)
                
            elif(k == 1): #SOUTH
                if(pos[0]+1 < 15):
                    arr.append(gridWorld.iloc[y+1][x])
                else:
                    arr.append(outArena)

            elif(k == 2): #EAST
                if(x+1 < 15):
                    arr.append(gridWorld.iloc[y][x+1])
                else:
                    arr.append(outArena)

            elif(k == 3): #WEST
                if(x-1 > -1):
                    arr.append(gridWorld.iloc[y][x-1])
                else:
                    arr.append(outArena)
                
        rTable.append(arr)

    return np.array(rTable)

def initQtable(gridWorld):
    qTable = []
    for x in range(len(gridWorld)**2):
        arr = []
        for k in range(4):
            arr.append(0)
        qTable.append(arr)
    return np.array(qTable)

def qVal(curState, action, maxAction, rTable):
    curRval = rTable.iloc[curState,action]
    value = curRval + (gamma * maxAction)
    return value

def maxValueQ(state, qTable):
    arr = []
    for i in qTable.iloc[state]:
        if(i != ''):
            arr.append(i)
        
    return max(list(map(float, arr)))
        
def actionPos(pos, world):
    _act =  random.randrange(0, 4)
    if(_act == 0 and pos[0]-1 < 0):
        return actionPos(pos, world)
    if(_act == 1 and pos[0]+1 > 14):
        return actionPos(pos, world)
    if(_act == 2 and pos[1]+1 > 14):
        return actionPos(pos, world)
    if(_act == 3 and pos[1]-1 < 0):
        return actionPos(pos, world)
    
    return _act

def state2Pos(state, lenQtable):
    _pos = [0,1]
    for i in range(lenQtable):
        if(i == state):
            return _pos
        else:
            if(i != 0):
                if(_pos[1] == 14):
                    _pos[0] += 1
                    _pos[1] = 0 
                else:
                    _pos[1] += 1

def pos2State(pos, gridWorld):
    state = 0
    for i in range(len(gridWorld)):
        for j in range(len(gridWorld[i])):
            if(pos[0] == i and pos[1] == j):
                return state
            state+=1

def step(currentPos, _dir):
    if(_dir == 0): 
        if(currentPos[0]-1 == -1):
            return step(currentPos, random.randrange(0, 4))
        currentPos[0]-=1
    if(_dir == 1): 
        if(currentPos[0]+1 == 15):
            return step(currentPos, random.randrange(0, 4))
        currentPos[0]+=1
    if(_dir == 2): 
        if(currentPos[1]+1 == 15):
           return step(currentPos, random.randrange(0, 4))
        currentPos[1]+=1
    if(_dir == 3): 
        if(currentPos[1]-1 == -1):
           return step(currentPos, random.randrange(0, 4))
        currentPos[1]-=1
    
    return currentPos

def stepBasedQ(state):
    arrDir = []    
    for i in range(len(_qTable.iloc[state])):
        arrDir.append([_qTable.iloc[state, i], i])
    _dir = max(arrDir)[1]

    return step(currentPos, _dir)

_gridWorld = pd.read_csv('DataTugas3ML2019.txt', sep="\t", header=None)
_rTable = pd.DataFrame(initRtable(_gridWorld), columns=["North", "South", "East", "West"])
_qTable = pd.DataFrame(initQtable(_gridWorld), columns=["North", "South", "East", "West"])

episodes = 15
gamma = 0.9
currentPos = [14,0]
currentState = pos2State(currentPos, _gridWorld) 
futureState = currentState
print('Proses training dimulai.. ', episodes, ' Episode')
print('Episode :', end=' ')

for i in range(episodes):
    currentState = random.randrange(0,224)

    while(currentState != 14):
        pos = state2Pos(currentState, len(_qTable))
        action = actionPos(pos, _gridWorld)
        goto = step(pos, action)
        futureState = pos2State(goto, _gridWorld) 
        _qTable.iloc[currentState,action] = qVal(curState=currentState, action=action, maxAction=maxValueQ(futureState, _qTable), rTable=_rTable)
        currentState = futureState
    print(i, end=' ')
        
isFinish = False


with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print('Q Table : ')
    print(_qTable)

totReward = 0
countStep = 0
while(not isFinish):
    state = pos2State(currentPos, _gridWorld)
    countStep += 1
    print('pos \t', currentPos, 'state \t', state)
    totReward += _gridWorld.iloc[currentPos[0],currentPos[1]]
    currentPos = stepBasedQ(state)

    if(currentPos[0] == 0 and currentPos[1] == 14):
        totReward += _gridWorld.iloc[currentPos[0],currentPos[1]]
        print('FINISH, total reward : ', totReward, ', Step : ', countStep)
        isFinish = True