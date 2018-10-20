import numpy as np

class SOMnetwork(object):
  # Implementation of a Self Organizing Network (2D output)

  def __init__(self, num_rows, num_cols, lr0, lr_c0):
    # Initialization

    self.num_rows = num_rows
    self.num_cols = num_cols
    self.lr0 = lr0
    self.lr_c0 = lr_c0
    self.weights = None

  def fit(self, data, num_samples, num_features, adapt_iterations, converge_iterations):
    num_samples = data.shape[0]
    num_features = data.shape[1]
    num_clusters = self.num_rows * self.num_cols

    # set a random weight of neuron
    self.weights = np.random.rand(num_clusters, num_features)

    # define constants
    # radius of two dimensional lattice
    sigma0 = 0.5 * np.sqrt(self.num_rows**2 + self.num_cols**2)

    # time constant
    tau1 = adapt_iterations / np.log(sigma0)

    # Self organizing phase
    for i in range(adapt_iterations):
      distance_pool = []
      lr = self.lr0 * np.exp(-i / adapt_iterations)
      sigma = sigma0 * np.exp(-i / tau1)

      j = np.random.randint(0, num_samples)
      for k in range(num_clusters):
        distance_pool.append(np.linalg.norm(data[j,:] - self.weights[k, :]))

      min_d = np.argmin(distance_pool)    # index of winning neuron (flat array)
      index_row, index_col = min_d // self.num_rows, min_d % self.num_rows
      # print min_d, index_row, index_col

      space_distance = [np.sqrt((index_row-m)**2 + (index_col-k)**2) for m in range(self.num_rows) for k in range(self.num_cols)]
      # print np.array(space_distance).reshape(self.num_rows, self.num_cols)

      # update all neurons
      for k in range(num_clusters):
        nhf = np.exp(- space_distance[k]**2 / (2 * sigma**2))
        self.weights[k, :] = self.weights[k, :] + lr * nhf * (data[j] - self.weights[k, :])

    # Converging phase
    for i in range(converge_iterations):
      distance_pool = []

      j = np.random.randint(0, num_samples)
      for k in range(num_clusters):
        distance_pool.append(np.linalg.norm(data[j,:] - self.weights[k, :]))

      min_d = np.argmin(distance_pool)

      # update at winner neurons only
      self.weights[min_d, :] = self.weights[min_d, :] + self.lr_c0 * (data[j] - self.weights[min_d, :])

    # print self.weights[:, 0:2]

