import numpy as np 

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

    def viterbi(self,file):
        stateSeqs = []
        obsSeqs = []

        obsPerSentence = ['0']
        for line in open(file, "r"):
            if line != '\n':
                obsPerSentence.append(line.strip())
            else:
                obsPerSentence.append('-1')

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
                            arr.append(pi[prevState][position-1] * self.a[prevState][currState] * self.b[obsIndex][currState])
                        pi[currState][position] = np.max(arr)
                        parent[currState][position] = np.argmax(arr)
                
                stop_index = self.stateSet.index("STOP")
                arr = np.array([])
                for pS in range(numStates):
                    arr.append(pi[pS][numPositions-2] * self.a[pS][stop_index]) 
                pi[stop_index][numPositions-1] = np.max(arr)
                parent[stop_index][numPositions-1] = np.argmax(arr)

                # backtracking
                optimalStateIndex = stop_index
                optimalStatels = []
                for p in range(numPositions-1, 2, -1):
                    optimalStateIndex = parent[optimalStateIndex][p]
                    optimalStatels.append(self.stateSet.index(optimalStateIndex))
                
                stateSeqs.append(optimalStatels.reverse())
                obsSeqs.append(obsPerSentence[1:-1])

                obsPerSentence = ['0']
    
        return stateSeqs, obsSeqs

    def viterbi_top_k(self,file, k):
        stateSeqs = []
        obsSeqs = []

        obsPerSentence = ['0']
        for line in open(file, "r"):
            if line != '\n':
                obsPerSentence.append(line.strip())
            else:
                obsPerSentence.append('-1')

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
                optimalStateIndex, optimalKValueIndex = stop_index, 0
                optimalStatels = []
                for p in range(numPositions-1, 2, -1):
                    optimalStateIndex, optimalKValueIndex = parent[p][optimalStateIndex][optimalKValueIndex]
                    optimalStatels.append(self.stateSet.index(optimalStateIndex))
                
                stateSeqs.append(optimalStatels.reverse())
                obsSeqs.append(obsPerSentence[1:-1])

                obsPerSentence = ['0']

        return stateSeqs, obsSeqs
                
                
# model = HMM()
# model.fileToSentence('train')
# print('obs', model.trainObs[0:2])
# print('states', model.trainStates[0:2])