# CODE SOURCE IS DIRECTLY FROM DOCUMENTATION
# https://scikit-learn.org/stable/auto_examples/svm/plot_separating_hyperplane.html
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.inspection import DecisionBoundaryDisplay


def plot_svm_boundary(model,X,y,title):
    """
    Customizing Scikit developers' code with the assumption of only 2 columns/features.

    """
    
    X = X.values
    y = y.values
    
    # Scatter Plot
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30,cmap='seismic')

    
    # plot the decision function
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.decision_function(xy).reshape(XX.shape)

    # plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])
    # plot support vectors
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
               linewidth=1, facecolors='none', edgecolors='k')
    plt.title(title)
    plt.show()


def plot_svm_boundary_with_pca(model, X, y, title):
    """

    Customizing Scikit Code using Principal Component Analysis (PCA) to reduce the feature space to 2D 
    for visualization, while still training the SVM on the full feature set.

    Assumption is X can also be more features, not just 2 features.
    """
    # Convert to NumPy arrays
    X_np = X.values if hasattr(X, 'values') else X
    y_np = y.values if hasattr(y, 'values') else y

    # Apply PCA to reduce to 2D for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_np)

    # Scatter plot of PCA-reduced data
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_np, s=30, cmap='seismic')

    # Create grid for decision boundary
    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    xx = np.linspace(xlim[0], xlim[1], 100)
    yy = np.linspace(ylim[0], ylim[1], 100)
    YY, XX = np.meshgrid(yy, xx)
    grid = np.vstack([XX.ravel(), YY.ravel()]).T

    # Inverse transform grid back to original feature space
    grid_original = pca.inverse_transform(grid)

    # Evaluate decision function on original feature space
    Z = model.decision_function(grid_original).reshape(XX.shape)

    # Plot decision boundary and margins
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1],
               alpha=0.5, linestyles=['--', '-', '--'])

    # Project support vectors to PCA space
    sv_pca = pca.transform(model.support_vectors_)
    ax.scatter(sv_pca[:, 0], sv_pca[:, 1], s=100,
               linewidth=1, facecolors='none', edgecolors='k')

    plt.title(title)
    plt.show()


def plot_svm_boundary2(title): # original code from Scikit developers
    # Authors: The scikit-learn developers
    # SPDX-License-Identifier: BSD-3-Clause

    # we create 40 separable points
    X, y = make_blobs(n_samples=40, centers=2, random_state=6)

    # fit the model, don't regularize for illustration purposes
    clf = svm.SVC(kernel="linear", C=1000)
    clf.fit(X, y)

    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

    # plot the decision function
    ax = plt.gca()
    DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        plot_method="contour",
        colors="k",
        levels=[-1, 0, 1],
        alpha=0.5,
        linestyles=["--", "-", "--"],
        ax=ax,
    )
    # plot support vectors
    ax.scatter(
        clf.support_vectors_[:, 0],
        clf.support_vectors_[:, 1],
        s=100,
        linewidth=1,
        facecolors="none",
        edgecolors="k",
    )

    plt.title(title)
    plt.show()




