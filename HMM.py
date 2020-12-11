import numpy as np 
import os 

class HMM(object):

    def __init__(self):
            self.stateSet = ['START']
            self.obsSet = []
            self.a = None 
            self.b = None
            self.trainObs = []
            self.trainStates=[]
            
    def readfile(self, file):
        for line in open(file, "r"):
            if line != '\n':
                x, y = line.strip().split(' ')
                if x not in self.obsSet: 
                    self.obsSet.append(x)
                if y not in self.stateSet:
                    self.stateSet.append(y)
        self.stateSet.append('STOP')

    def fileToSentence(self, file):
        obs = []
        states = []
        obsPerSentence = []
        statesPerSentence = []
        for line in open(file, "r"):
            if line != '\n':
                o, s = line.strip().split(' ')
                obsPerSentence.append(o)
                statesPerSentence.append(s)
            else:
                self.trainObs.append(obsPerSentence)
                self.trainStates.append(statesPerSentence)
                obsPerSentence = []
                statesPerSentence = []
        return 

    def estimate_a(self, file):

        lsOfCurrNextStates = []

        for sentence in self.trainStates:
            sentence.insert(0,'START')
            sentence.append('STOP')
            for element in list(zip(sentence, sentence[1:])): 
                lsOfCurrNextStates.append(element)
        
        countCurrNext = [[0 for i in range(len(self.stateSet))] for i in range(len(self.stateSet))]
        countCurr = [0 for i in range(len(self.stateSet))]

        for curr, nex in lsOfCurrNextStates:
            currIndex = self.stateSet.index(curr.strip())
            nexIndex = self.stateSet.index(nex.strip())
            countCurrNext[currIndex][nexIndex] += 1
            countCurr[currIndex] += 1
        
        for i in range(len(self.stateSet)):
            for j in range(len(self.stateSet)):
                if countCurr[i] == 0:
                    countCurrNext[i][j] = 0 
                else: 
                    countCurrNext[i][j] = countCurrNext[i][j]/countCurr[i]
        
        self.a = countCurrNext
        
        return

    def improved_estimate_b(self, file, k=0.5):
        #add 'unknown' label 
        self.obsSet.append("#UNK#")

        y = []
        y_to_x = [[0 for i in range(len(self.stateSet))] for i in range(len(self.obsSet))]
        output = [[0 for i in range(len(self.stateSet))] for i in range(len(self.obsSet))]

        #get the count(y->x) for each y and x. Basically count(y->x)=y_to_x[x_index][y_index]
        for line in open(file, "r"):
            if line != '\n':
                x, y = line.split(' ')
                y_index = self.stateSet.index(y.strip())
                try: 
                    x_index = self.obsSet.index(x.strip())
                except ValueError: 
                    x_index = self.stateSet.index('#UNK#')
                y_to_x[x_index][y_index] += 1
        
        #count(y) = y[y_index]
        y =  [sum(i) for i in zip(*y_to_x)] 
        unknown_x = self.obsSet.index("#UNK#")
        
        for i in range(len(self.obsSet)):
            for j in range(len(self.stateSet)):
                if (y[j]+k) == 0 or y[j] == 0: 
                    output[i][j] = 0
                else: 
                    if i == unknown_x: 
                        output[i][j] = k/(y[j]+k)
                    else: 
                        output[i][j] = y_to_x[i][j]/(y[j]+k)
        self.b = output
        return self.b 

    def viterbi(self, obsPerSentence):

        numPositions = len(obsPerSentence)
        numStates = len(self.stateSet)
        pi = [[0 for i in range(numPositions)] for i in range(numStates)]
        parent = [[0 for i in range(numPositions)] for i in range(numStates)]
        
        pi[0][0] = 1 

        # pi VS a,b index 
        # based on pi index 
        for position in range(1, numPositions-1): 
            try: 
                obsIndex = self.obsSet.index(obsPerSentence[position])
            except ValueError:
                obsIndex = -1 
            for currState in range(numStates):
                arr = np.array([])
                for prevState in range(numStates):
                    arr = np.append(arr, pi[prevState][position-1] * self.a[prevState][currState] * self.b[obsIndex][currState])
                pi[currState][position] = np.max(arr)
                parent[currState][position] = np.argmax(arr)
        
        stop_index = self.stateSet.index("STOP")
        arr = np.array([])
        for pS in range(numStates):
            arr = np.append(arr, pi[pS][numPositions-2] * self.a[pS][stop_index]) 
        pi[stop_index][numPositions-1] = np.max(arr)
        parent[stop_index][numPositions-1] = np.argmax(arr)

        # backtracking
        optimalStateIndex = stop_index
        optimalStatels = []
        for p in range(numPositions-1, 2, -1):
            optimalStateIndex = parent[optimalStateIndex][p]
            optimalStatels.append(self.stateSet[optimalStateIndex])
        
        optimalStatels.reverse()
        obsPerSentence = obsPerSentence[1:-1]

        return optimalStatels, obsPerSentence

    def viterbi_top_k(self, k, obsPerSentence):

        numPositions = len(obsPerSentence)
        numStates = len(self.stateSet)
        pi = [[[0 for i in range(k)] for i in range(numStates)] for i in range(numPositions)]
        parent = [[[0 for i in range(k)] for i in range(numStates)] for i in range(numPositions)]

        pi[0][0] = [1]*k

        for position in range(1, numPositions-1):
            try: 
                obsIndex = self.obsSet.index(obsPerSentence[position])
            except ValueError:
                obsIndex = -1 
            for currState in range(numStates):
                ls = []
                for prevState in range(numStates):
                    for top in range(k): 
                        ls.append(pi[position-1][prevState][top] * self.a[prevState][currState] * self.b[obsIndex][currState])
                topKls = sorted(ls, reverse=True)[:3]
                pi[position][currState] = topKls 
                for top in range(k):
                    indexInls = ls.index(topKls[top])
                    parentState = indexInls//k 
                    parentKValue = indexInls - parentState*k
                    parent[position][currState][top] = (parentState, parentKValue)

        stop_index = self.stateSet.index("STOP")
        ls = []
        for prevState in range(numStates):
            for top in range(k): 
                ls.append(pi[numPositions-2][prevState][top] * self.a[prevState][stop_index])
        topKls = sorted(ls, reverse=True)[:3]
        pi[numPositions-1][stop_index] = topKls 
        for top in range(k):
            indexInls = ls.index(topKls[top])
            parentState = indexInls//k 
            parentKValue = indexInls - parentState*k
            parent[numPositions-1][stop_index][top] = (parentState, parentKValue)        

        # backtracking
        optimalStateIndex, optimalKValueIndex = stop_index, 2
        optimalStatels = []
        for p in range(numPositions-1, 2, -1):
            optimalStateIndex, optimalKValueIndex = parent[p][optimalStateIndex][optimalKValueIndex]
            optimalStatels.append(self.stateSet[optimalStateIndex])

        optimalStatels.reverse()
        obsPerSentence = obsPerSentence[1:-1]

        return optimalStatels, obsPerSentence



    def predictGlobal(self, file, k):

        f = open("EN/dev.p3.out", "w")

        obsPerSentence = ['0']
        for line in open(file, "r"):
            if line != '\n':
                obsPerSentence.append(line.strip())
            else:
                obsPerSentence.append('-1')

                if k == 1:
                    optimalStatels, obsPerSentence = self.viterbi(obsPerSentence)
                    print('optimalStatels',optimalStatels)
                    print('obsPerSentence', obsPerSentence)
                else: 
                    optimalStatels, obsPerSentence = self.viterbi_top_k(obsPerSentence, k)
                
                
                for s, ob in zip(optimalStatels, obsPerSentence):
                    f.write("{0} {1}\n".format(ob, s))
                    f.write("\n")
                
                obsPerSentence = ['0']

        f.close()

        return 


                
# model = HMM()
# model.fileToSentence('EN/train')
# print('obs', model.trainObs[0:2])
# print('states', model.trainStates[0:2])

# trainFilePath = ['EN/train', 'SG/train', 'CH/train']
# testFilePath = ['EN/dev.in', 'SG/dev.in', 'CH/dev.in']
# trueFilePath = ['EN/dev.out', 'SG/dev.out', 'CH/dev.out']


model = HMM()

#initialise self.stateSet and self.obsSet
model.readfile('EN/train')
#initialise self.trainObs and self.trainStates
model.fileToSentence('EN/train')

# initialise self.a and self.b 
model.estimate_a('EN/train')
# print('a', model.a[:4][:4])
model.improved_estimate_b('EN/train')
# print('b', model.b[:4][:4])

model.predictGlobal('EN/dev.in', 1)


    