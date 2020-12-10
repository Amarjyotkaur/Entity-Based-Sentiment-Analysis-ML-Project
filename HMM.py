import numpy as np 

class HMM(object):

    def __init__(self):
            self.stateSet = ['START']
            self.obsSet = []
            self.a = None 
            self.b = None
            self.trainObs = []
            self.trainStates=[]

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

    # # To be modified after part 4 
    # def predictSeq(self, file, algorithm):
    #     obsPerSentence = []
    #     for line in open(file, "r"):
    #         if line != '\n':
    #             obsPerSentence.append(line.strip())
    #         else:
    #             if algorithm = "Viterbi":
    #                 Viterbi()


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
                for position in range(numPositions-1): #start from 1 
                    try: 
                        obsIndex = self.obsSet.index(obsPerSentence[position])
                    except ValueError:
                        obsIndex = -1 
                    for currState in range(numStates):
                        arr = np.array([])
                        for prevState in range(numStates):
                            arr.append(pi[prevState][position-1] * self.a[prevState][currState] * self.b[obsIndex][currState])
                        pi[currState][position] = max(arr)
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
                
                stateSeqs.append(optimalStatels)
                obsSeqs.append(obsPerSentence[1:-2])
    
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

                for position in range(numPositions-1):
                    try: 
                        obsIndex = self.obsSet.index(obsPerSentence[position])
                    except ValueError:
                        obsIndex = -1 
                    for currState in range(numStates):
                        arr = np.array([])
                        for prevState in range(numStates):
                            arr.append(pi[prevState][position-1] * self.a[prevState][currState] * self.b[obsIndex][currState])
                        pi[currState][position] = max(arr)
                        parent[currState][position] = np.argmax(arr)




# model = HMM()
# model.fileToSentence('train')
# print('obs', model.trainObs[0:2])
# print('states', model.trainStates[0:2])
