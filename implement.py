import numpy as np
import csv
from rbf import RBFnetwork

# get inputs from csv files

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
# try randomly pick, implement the rest later

model = RBFnetwork(input_shape=len(training_data[0]), hidden_neurons=20, sigma=3.05)
model.fit(training_data, training_label)

model_out = model.predict(training_data)
model_out = np.array(model_out)

# print(min(model_out))
# print(max(model_out))

print(model_out)

accuracy = model.accuracy(model_out, training_label)
print('accuracy: ', accuracy)

# training_data = [[1, 1], [0, 1], [0, 0], [1, 0]]
# training_label = [0, 1, 0, 1]

# training_data = np.array(training_data)
# training_label = np.array(training_label)

# model = RBFnetwork(input_shape=len(training_data[0]), hidden_neurons=2, sigma=0.707)
# model.fit(training_data, training_label)

# model_out = model.predict(training_data)
# print(model_out)
