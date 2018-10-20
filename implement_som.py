import numpy as np
import csv
from som import SOMnetwork
from sklearn import datasets

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

model = SOMnetwork(num_rows=3, num_cols=3, lr0=0.10, lr_c0=0.01)
model.fit(training_data, len(training_data), len(training_data[0]), 65000, 18000)

# x = datasets.load_iris().data

# model = SOMnetwork(num_rows=3, num_cols=3, lr0=0.1, lr_c0=0.01)
# model.fit(x, len(x), len(x[0]), 500, 1000)
