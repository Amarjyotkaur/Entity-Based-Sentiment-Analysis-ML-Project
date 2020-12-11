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

    #Note: for the files add two more empty lines at the end or else this function won't work properly lol
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

    def estimate_b(self, file): 
        y = []
        y_to_x = [[0 for i in range(len(self.stateSet))] for i in range(len(self.obsSet))]
        for line in open(file, "r"):
            if line != '\n':
                x, y = line.split(' ')
                y_index = self.stateSet.index(y.strip())
                x_index = self.obsSet.index(x.strip())
                y_to_x[x_index][y_index] += 1
        
        y =  [sum(i) for i in zip(*y_to_x)] 
        
        for i in range(len(self.obsSet)):
            for j in range(len(self.stateSet)):
                if y[j] == 0: 
                    y_to_x[i][j] = 0
                else: 
                    y_to_x[i][j] = y_to_x[i][j]/y[j]
        self.b = y_to_x
        return self.b

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

    def get_tag(self, x): 
        if x not in self.obsSet:
            x = "#UNK#" 
        x_index = self.obsSet.index(x)

        #retrive the list of probabilities for the input observation
        prob_for_each_x = self.b[x_index]
        max_y = max(prob_for_each_x) 
        max_y_index = prob_for_each_x.index(max_y)
        #return tag 
        return self.stateSet[max_y_index]

    def predict(self, test_file):
        f = open("dev.p2.out", "w")
        for line in open(test_file, "r"):
            if line != "\n":
                x = line.strip()
                f.write("{0} {1}\n".format(x, self.get_tag(x)))
            else: 
                f.write(line)

        f.close()

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
    
    #TODO this does not look right
    # ignore hashtags 
    # two metrics: entity and sentiment 
    #so basically the confusion matrix will be 2x2 yay now correct liao 
    def evaluate2(self, pred_file, actual_file):
        
        pass
    def evaluate(self, pred_file, actual_file):
        y_pred = []
        y_actual = []
        for line in open(actual_file, "r"):
            if line != "\n":
                x, y = line.strip().split(" ")
                y_actual.append(y)
        for line in open(pred_file, "r"):
            y_pred = []
            if line != "\n":
                x, y = line.strip().split(" ")
                y_pred.append(y)
        confusion_matrix = [[0 for i in self.stateSet] for i in self.stateSet] 
        for i in range(len(y_pred)):
            pred_index = self.stateSet.index(y_pred[i])
            actual_index = self.stateSet.index(y_actual[i])
            confusion_matrix[actual_index][pred_index] += 1
        
        true_positive = [confusion_matrix[i][i] for i in range(len(self.stateSet))]
        false_positive = [0 for i in range(len(self.stateSet))]
        false_negative = [0 for i in range(len(self.stateSet))]
        precision = recall = [0 for i in range(len(self.stateSet))]

        for correct in range(len(self.stateSet)):
            for i in range(len(self.stateSet)):
                if i != correct: 
                    false_positive[correct] += confusion_matrix[i][correct]
        
        for correct in range(len(self.stateSet)):
            for i in range(len(self.stateSet)):
                if i != correct: 
                    false_negative[correct] += confusion_matrix[correct][i]


        for i in range(len(true_positive)):
            if (true_positive[i] + false_positive[i]) == 0: 
                precision[i] = 0
            else: 
                try: 
                    precision[i] = true_positive[i]/(true_positive[i] + false_positive[i])
                except ZeroDivisionError: 
                    precision[i] = 0
        precision = sum(precision)

        for i in range(len(true_positive)):
            if (true_positive[i] + false_negative[i]) == 0: 
                recall[i] = 0
            else: 
                try: 
                    recall[i] = true_positive[i]/(true_positive[i] + false_negative[i])
                except ZeroDivisionError: 
                    recall[i] = 0
        recall = sum(recall)
        
        # f1 = 2/((1/precision) + 1/recall)
        print("Precision: ", precision)
        print("Recall: ", recall)
        # print("F1 score: ", f1)
        print(confusion_matrix[3])




model = HMM()
#initialise self.stateSet and self.obsSet
model.readfile('./CN/train')
#initialise self.trainObs and self.trainStates
model.fileToSentence('./CN/train')
model.improved_estimate_b('./CN/train')

# print("Observation set: {0}\n".format(model.obsSet))
# print("State set: {0}\n".format(model.stateSet))
# print("Train Obs: {0}\n".format(model.trainObs))
# print("Train States: {0}\n".format(model.trainStates))
# print("b value: {0}\n".format(model.b))
model.predict('./CN/dev.in')
# model.get_tag("HBO")


# model.evaluate("dev.p2.out", "./SG/dev.out")
# print('observation set', model.obsSet[0:10])
# print('stateset', model.stateSet)
# print('obs', model.trainObs[0:2])
# print('states', model.trainStates[0:2])
