import numpy as np
from collections import Counter

def euclidean_distance(a, b):
    """Compute the euclidean distance between two numpy arrays.

    Parameters
    ----------
    a: numpy array
    b: numpy array

    Returns
    -------
    distance: float
    """
    return np.linalg.norm(a-b) 




class KNNClassifier:
    """Regressor implementing the k-nearest neighbors algorithm.

    Parameters
    ----------
    k: int, optional (default = 5)
        Number of neighbors that are included in the prediction.
    distance: function, optional (default = euclidean)
        The distance function to use when computing distances.
    """

    def __init__(self, k=5, distance=euclidean_distance):
        """Initialize a KNNRegressor object."""
        self.k = k
        self.distance = distance

    def fit(self, X, y):
        """Fit the model using X as training data and y as target values.

        According to kNN algorithm, the training data is simply stored.

        Parameters
        ----------
        X: numpy array, shape = (n_observations, n_features)
            Training data.
        y: numpy array, shape = (n_observations,)
            Target values.

        Returns
        -------
        self
        """
        self.X = X
        self.y = y
        return self

    def get_all_distances_from_point (self,p):
        return np.array([self.distance(x, p) for x in self.X])

    def predict(self, X):
        """Return the predicted values for the input X test data.

        Assumes shape of X is [n_test_observations, n_features] where
        n_features is the same as the n_features for the input training
        data.

        Parameters
        ----------
        X: numpy array, shape = (n_observations, n_features)
            Test data.

        Returns
        -------
        result: numpy array, shape = (n_observations,)
            Predicted values for each test data sample.

        """
        def Most_Common(lst):
            data = Counter(lst)
            return data.most_common(1)[0][0]

        predictions = []
        for x in X:
            dists = self.get_all_distances_from_point( x)
            nearest_k = self.y[dists.argsort()[:self.k]]
            mc_list = Counter(nearest_k).most_common(3)
            most_common_count = mc_list[0][1]
            most_common_values = [mc[0] for mc in mc_list if mc[1] == most_common_count]
  #    print(nearest_k, most_common_values)
            predictions.append(np.random.choice(most_common_values, size=1)[0])
        return np.array(predictions)


        
            



#if __name__ == '__main__':

   # X, y = make_data(n_features=2, n_pts=300, noise=0.1)
    #y = y.reshape(30)
  #  model = KNNRegressor(distance=cosine_distance)
  #  model.fit(X, y)
    #predictions = model.predict(X)
  #  print(model.predict([[0,0]]))
 #   fig, ax = plt.subplots()
  #  plot_predictions(ax, model, X, y)
  #  fig.show()