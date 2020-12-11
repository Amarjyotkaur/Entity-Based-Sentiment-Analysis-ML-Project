from datetime import datetime
import numpy as np 

class MultiClassPerceptron(object):
    def __init__(self):
        self.uniqueWords = []
        self.uniqueClasses = []
        self.sentences = []
        self.theta = None #this is supposed to be a nested array
        self.x_train = None 
        self.y_train = None 

    def getOneHotEncoding(self, index):
        return np.array([0 if j != index else 1 for j in range(len(self.uniqueWords))]) 
        
    def setUniqueXandY(self, train_file):
        for line in open(train_file, 'r'):
            if line !='\n':
                x, y = line.strip().split(' ')
                if x not in self.uniqueWords: 
                    self.uniqueWords.append(x)
                if y not in self.uniqueClasses: 
                    self.uniqueClasses.append(y)
        self.uniqueWords.append("#UNK#")
    def readTrainFile(self, train_file):
        self.x_train = []
        self.y_train = []

        self.setUniqueXandY(train_file)
        
        #get my x_train and y_train 
        for line in open(train_file, 'r'):
            if line != '\n':
                x, y = line.strip().split(' ')
                self.x_train.append(x)
                self.y_train.append(y)

    def train(self, n_epochs):
        #initialise theta to 0 
        self.theta = [[0 for i in self.uniqueWords] for i in self.uniqueClasses]
        if len(self.x_train) != len(self.y_train):
            raise ValueError('x_train and y_train have different shape!')
        
        for n in n_epochs: 
            for i in range(len(self.x_train)): 
                print('Line {0}\n'.format(i))
                #get the one-hot encoding of each x 
                x_i = self.x_train[i]
                x_index = self.uniqueWords.index(x_i)
                x_vector = self.getOneHotEncoding(x_index)
                #getting the maximum y for all possible classes
                y_pred_val = [np.dot(self.theta[j], x_vector) for j in range(len(self.uniqueClasses))]
                max_y_index = y_pred_val.index(max(y_pred_val)) 

                #if prediction is wrong, update the weights 
                if self.uniqueClasses[max_y_index] != self.y_train[i]:
                    #subtract weights of the predicted y 
                    print('Original w*', self.theta[max_y_index])
                    self.theta[max_y_index] = np.subtract(self.theta[max_y_index], x_vector) 
                    print('new w*', self.theta[max_y_index])
                    #add weights of the correct y 
                    actual_y_index = self.uniqueClasses.index(self.y_train[i])
                    self.theta[actual_y_index] = np.add(self.theta[actual_y_index], x_vector)
    
    def predict(self, test_file):
        f = open('dev.p3.out', 'w')
        for line in open(test_file, 'r'):
            if line != '\n':
                x_test = line.strip()
                try: 
                    x_index = self.uniqueWords.index(x_test)
                except ValueError: 
                    x_index = self.uniqueWords.index("#UNK#")
                
                x_vector = self.getOneHotEncoding(x_index)

                y_pred_val = [np.dot(self.theta[j], x_vector) for j in range(len(self.uniqueClasses))] 
                max_y_index = y_pred_val.index(max(y_pred_val))
                argmax_y = self.uniqueClasses[max_y_index] 
                f.write("{0} {1}\n".format(x_test, argmax_y))
            else: 
                f.write(line)
        f.close()
    
start = datetime.now()
model = MultiClassPerceptron()
model.readTrainFile('./SG/train')
print('initialised!')
model.train(1)
print('trained!')
model.predict('./SG/dev.in')
end = datetime.now()
print('Time: ', (end-start).total_seconds())