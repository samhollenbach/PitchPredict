# Created by Andy Hayes and Sam Hollenbach for the COMP 484 Final Project

import numpy as np
import csv

# X = (hours sleeping, hours studying), y = score on test
# X = np.array(([1, .5, 3, 1, .275], [0, 1, 3, 1, .260], [3, .33, 2, 0, .300], [0, .0, 1, 0, .300],
#               [2, 0.66, 4, 1, .237], [1, 0.8, 4, 0, .250]), dtype=float)
# y = np.array(([1], [1], [0], [1], [1], [0]), dtype=float)

xPredicted = np.array(([3,2,0,6,7,1]), dtype=float) # Pitch situation we are predicting for
xPredicted2 = np.array(([0,0,0,1,0,0]), dtype=float)

pitchdat = [] # Create empty array to populate with pitch situation data vectors

resultdat = [] # Create empty array to store pitch results associated with the pitch situation input vectors

predictions = [] # Empty array to populate with the predictions from the

#function to read in the data file and populate arrays with associated pitch vectors
def createData():
    with open('sampledpitches2017.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:
            if line_count > 10000:
                return
            if line_count <= 1:
                line_count += 1
            else:
                pitch = list(row[1:7])
                pitchdat.append(pitch)
                resultdat.append(list(row[7]))
                line_count += 1

        print('done with file.')

#X = X/np.amax(X, axis=0)
# xPredicted = xPredicted/np.amax(xPredicted, axis=0) # maximum of xPredicted (our input data for the prediction)
# y = y/1

class Neural_Network(object):
  def __init__(self):
    #parameters
    self.inputSize = 6
    self.outputSize = 1
    self.hiddenSize = 6

    #weights
    self.W1 = np.random.randn(self.inputSize, self.hiddenSize) # weight matrix from input to hidden layer
    self.W2 = np.random.randn(self.hiddenSize, self.outputSize) # weight matrix from hidden to output layer

  def sigmoid(self, s):
    # sigmoid function for forward propagation
    signal = np.clip( s, -500, 500 )
    signal = 1.0 / (1 + np.exp(-s))
    return signal

  def sigmoidPrime(self, s):
      # derivative of sigmoid = rate of change for the activation function used for backwards propagation
      return s * (1 - s)

  def forward(self, X):
    #function to handle forward propagation
    self.z = np.dot(X, self.W1) # dot product of X (input) and first set of weights
    self.z2 = self.sigmoid(self.z) # activation function
    self.z3 = np.dot(self.z2, self.W2) # dot product of hidden layer (z2) and second set of weights
    o = self.sigmoid(self.z3)# final activation function
    return o

  def backward(self, X, y, o):
      # backward propagate funciton, were tryin to get better!!
      self.o_error = y - o  # error in output
      self.o_delta = self.o_error * self.sigmoidPrime(o)  # its about the direction babe, cmon shoot for the root

      self.z2_error = self.o_delta.dot(self.W2.T)  # how bad our weights were to the error
      self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)  # applying derivative of sigmoid to z2 error

      self.W1 += X.T.dot(self.z2_delta)  # fix inner weights
      self.W2 += self.z2.T.dot(self.o_delta) # fix outer weights

  def predict(self, X):
      print("Predicted data based on trained weights: ")
      print("Output: \n " + self.forward(X))
      # if self.forward((xPredicted) >= .5):
      #     print("fastball")
      # else:
      #     print("offspeed")

  def train(self, X, y):
      o = self.forward(X)
      self.backward(X, y, o)
      predictions.append(o)





if __name__ == '__main__':
    # NN = Neural_Network()
    # # defining our output
    # o = NN.forward(X)
    # print("Predicted Output: \n" + str(o))
    # print("Actual Output: \n" + str(y))
    createData()

    X = np.array((pitchdat), dtype=float)
    y = np.array((resultdat), dtype=float)
    #xPredicted = np.array(([3,2,0,6,7,1]), dtype=float)

    X = X / np.amax(X, axis=0)
    xPredicted = xPredicted / np.amax(xPredicted, axis=0)  # maximum of xPredicted (our input data for the prediction)
    y = y


    NN = Neural_Network()
    NN.train(X, y)
    #for i in range(1):  # trains the NN 1,000 times
        # print("Input: \n" + str(X))
        # print("Actual Output: \n" + str(y))
        # #print("Predicted Output: \n" + str(NN.forward(X)))
        # print("Loss: \n" + str(np.mean(np.square(y - NN.forward(X)))))  # mean sum squared loss
        #print("training\n")
        #NN.train(X, y)

    print(NN.forward(xPredicted))
    #NN.predict(np.array(([3,2,0,6,7,1]), dtype=float))



