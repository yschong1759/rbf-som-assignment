import numpy as np
import csv
from rbf import RBFnetwork
from som import SOMnetwork
import time

# get inputs from csv files

print "Initializing..."

training_data_filename = 'data/data_train.csv'
training_label_filename = 'data/label_train.csv'

training_data = []
training_label = []

with open(training_data_filename) as csvfile:
  reader = csv.reader(csvfile, delimiter=',')
  for row in reader:
    for i in range(len(row)):
      row[i] = float(row[i])
    training_data.append(row)

with open(training_label_filename) as csvfile:
  reader = csv.reader(csvfile, delimiter=',')
  for row in reader:
    for i in range(len(row)):
      row[i] = float(row[i])
    training_label.append(row)

# use numpy format to represent the inputs
training_data = np.array(training_data)
training_label = np.array(training_label)

# select RBF center neurons, there are several options
# such as randomly pick, by k-means, or SOM
# here we use SOM network to pick for us

print "Building SOM Network to calculate center neurons for RBF network"

# initialize SOM network
model_som = SOMnetwork(num_rows=4, num_cols=5, lr0=0.10, lr_c0=0.01)
model_som.fit(training_data, len(training_data), len(training_data[0]), 65000, 18000)

print "Obtained center neurons!!"
print "Building RBF network..."

# initialize RBF network
model = RBFnetwork(input_shape=len(training_data[0]), hidden_neurons=20, sigma=3.05)
model.fit(training_data, training_label, 0, model_som.weights)

print "RBF network is set up..."
print "Predicting the output..."

model_out = model.predict(training_data)
model_out = np.array(model_out)

# Calculate the network accuracy
accuracy = model.accuracy(model_out, training_label)
print "Accuracy of overall network: ", accuracy, "%"

# Check performance with test data
print "Predicting the classes for each in test data"

# get inputs from csv files
predict_filename = 'data/data_test.csv'

testing_data = []

with open(predict_filename) as csvfile:
  reader = csv.reader(csvfile, delimiter=',')
  for row in reader:
    for i in range(len(row)):
      row[i] = float(row[i])
    testing_data.append(row)

testing_data = np.array(testing_data)

# Predict test data's output
predict_out = model.predict(testing_data)
predict_class = model.classify(predict_out, training_label)


print "Outputting the predicted labels"

print "\n", predict_class, "\n"

print "\n", np.unique(predict_class, return_counts=True)[1], "of the samples are of class", np.unique(predict_class, return_counts=True)[0], "respectively"

