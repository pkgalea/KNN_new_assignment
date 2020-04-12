import numpy as np
import matplotlib.pyplot as plt

def plot_predictions(ax, X_train, y_train, classifier = None, X_test=None, y_test=None): 
    """Plot the decision boundary of a kNN classifier.

    Plots predictions as colors.

    Assumes classifier, has a .predict() method that follows the
    sci-kit learn functionality.

    X_train and X_test must contain only 2 continuous features.

    Function modeled on scikit-learn example.

    Colors have been chosen for accessibility.


    Parameters
    ----------
    ax: a pyplot axis.  Call fig,ax = plt.subplots() and then pass in the axis to this function
    X_train: numpy array, shape = [n_observations, n_features]
        Training data to display.
    y_train: numpy array, shape = [n_observations,]
        Target labels.)
    X_test: optional
    classifier: optional instance of classifier object A fitted classifier with a .predict() method
    
    """
    mesh_count = 100.
    buffer = 0.05

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, m_max]x[y_min, y_max].

    feature_1 = X_train[:, 0]
    feature_2 = X_train[:, 1]
    x_min, x_max = feature_1.min(), feature_1.max()
    y_min, y_max = feature_2.min(), feature_2.max()
    v_min, v_max = y_train.min(), y_train.max()

    x_mesh_step_size = (x_max - x_min)/mesh_count
    y_mesh_step_size = (y_max - y_min)/mesh_count

    x_dist = x_max - x_min
    y_dist = y_max - y_min
    xx, yy = np.meshgrid(np.arange(x_min - x_dist * buffer , x_max  + x_dist * buffer, x_mesh_step_size),
                         np.arange(y_min - y_dist * buffer,  y_max + y_dist * buffer, y_mesh_step_size))
    
    if (classifier):
        values = classifier.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        values = values.reshape(xx.shape)
        ax.pcolormesh(xx, yy,
                  values,
                  cmap='Set3',
                  vmin=v_min,
                  vmax=v_max)
    
    colors = ["#0099FF", "#99FF00", "#FFFF00"]
    labels = ["Kama", "Rosa", "Canadian"]
    if (X_test!=y_test):
        for i in range(1,4):
            sctr = ax.scatter(X_test[:,0][y_test==i], X_test[:,1][y_test==i],
                          c=colors[i-1],
                          label=labels[i-1],
                          edgecolor='black', lw=0.4)
    else:
    
        # Plot the training points, saving the colormap
        for i in range(1,4):
            sctr = ax.scatter(feature_1[y_train==i], feature_2[y_train==i],
                          c=colors[i-1],
                          cmap='Set1',
                          edgecolor='black', lw=0.4, label=labels[i-1])
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xlabel('compactness')
    ax.set_ylabel('groove length')
    #fig.colorbar()
    ax.legend()
    if (classifier):
        ax.set_title("Classification predictions (k = {0}, metric = '{1}')"
                         .format(classifier.k, classifier.distance.__name__))
    else:
        ax.set_title ("Wheat Types")