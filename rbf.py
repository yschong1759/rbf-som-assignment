import numpy as np
from sklearn.cluster import KMeans

class RBFnetwork(object):
  # Implementation of a Radial Basis Function Network

  def __init__(self, input_shape, hidden_neurons, sigma=0.5):
    # Initialize

    # Arguments:
    #   input_shape: dimension of the input data
    #   hidden_neurons: number of hidden neurons
    #   sigma: tune the width of Gaussian RBF function

    self.input_shape = input_shape
    self.hidden_neurons = hidden_neurons
    self.sigma = sigma
    self.centers = []   # to store center neurons
    self.weights = None   # to store weights between hidden layer
                          # and output layer

  def kmeans(self, data, label, num_neurons):
    # k-means method for calculating neuron centers
    selected_centers = []

    for a_class in np.unique(label):
      cluster_members = []
      for element in np.argwhere(label==a_class):
        cluster_members.append(data[element[0]])

      kmeans = KMeans(n_clusters=num_neurons, max_iter=100)
      # fit kmeans object to data
      kmeans.fit(cluster_members)

      selected_centers.extend(kmeans.cluster_centers_)

    return selected_centers

  def _rbf_gauss(self, data, center):
    # Gaussian RBF Function
    data = np.array(data)
    center = np.array(center)
    return np.exp(np.linalg.norm(data - center)**2 / (-2 * self.sigma**2))

  def _calc_activation(self, data):
    # Calculate the activation signal at each center neurons
    signal_out = np.zeros((data.shape[0], self.hidden_neurons))

    # Iterate through all inputs
    for i in range(len(data)):
      for j in range(self.hidden_neurons):
        signal_out[i, j] = self._rbf_gauss(data[i], self.centers[j])

    return signal_out

  def _lin_reg_method(self, X, Y):
    # Perform linear regression
    return np.dot(np.linalg.pinv(X), Y)

  def fit(self, data, label, custom_centers, neuron_centers=None):
    # Get neuron centers
    # use provided neuron centers
    # otherwise use k-means to determine

    if custom_centers:
      # calculated from SOM network
      self.centers = np.array(neuron_centers)
    else:
      # Use k-means
      self.centers.extend(self.kmeans(data, label, self.hidden_neurons//2))
      # print('Length of center neurons', len(self.centers))

    # Get RBF functions, here we use Gaussian
    # Then calculate RBF activation signals

    rbf_signal = self._calc_activation(data)

    # add bias term
    rbf_signal = np.insert(rbf_signal, len(rbf_signal[0]), 1.0, axis=1)

    # Calculate weights from RBF function to output layer
    # Using linear regression method
    self.weights = self._lin_reg_method(rbf_signal, label)

  
  def predict(self, data):
    # Calculate the outputs of the RBF network

    signal_out = self._calc_activation(data)

    # add bias term
    signal_out = np.insert(signal_out, len(signal_out[0]), 1.0, axis=1)
    
    predictions = np.dot(signal_out, self.weights)
    
    return predictions

  def accuracy(self, out, desired):
    # Calculate the accuracy of RBF network
    diff = 0.9
    total_samples = len(desired)
    good_estimate = 0
    
    for i in range(total_samples):
      if(abs(out[i] - desired[i]) <= diff):
        good_estimate += 1


    return float(good_estimate) / total_samples * 100

  def classify(self, data, labels):
    # Classify the input data based on their outputs
    diff = 0.9
    total_outputs = len(data)
    output = []
    
    unique = np.unique(labels)

    for i in range(total_outputs):
      for label in unique:
        if(abs(label - data[i]) <= diff):
          output.append(label)

    return output
