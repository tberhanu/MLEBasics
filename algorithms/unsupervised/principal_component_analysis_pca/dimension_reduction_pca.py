import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def manual_pca():
    df = pd.read_csv('/Users/tess/Desktop/MLE2025/ML-Masterclass/UNZIP_FOR_NOTEBOOKS_FINAL (1)/DATA/cancer_tumor_data_features.csv')

                        ## Manual Construction of PCA

    scaler = StandardScaler()

    scaled_X = scaler.fit_transform(df)

    # Because we scaled the data, this won't produce any change.
    # We've left if here because you would need to do this for unscaled data
    scaled_X -= scaled_X.mean(axis=0)
    # Grab Covariance Matrix
    covariance_matrix = np.cov(scaled_X, rowvar=False)
    # Get Eigen Vectors and Eigen Values
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)

    # Choose number of components, 2 for simplicity of visualization
    num_components=2

    # Get index sorting key based on Eigen Values
    sorted_key = np.argsort(eigen_values)[::-1][:num_components]

    # Get num_components of Eigen Values and Eigen Vectors
    eigen_values, eigen_vectors = eigen_values[sorted_key], eigen_vectors[:, sorted_key]
    print("eigen_values", eigen_values)
    print("eigen_vectors", eigen_vectors)

    # Dot product of original data and eigen_vectors are the principal component values
    # This is the "projection" step of the original points on to the Principal Component

    principal_components=np.dot(scaled_X,eigen_vectors)
    print("principal_components", principal_components)

    plt.figure(figsize=(8,6))
    plt.scatter(principal_components[:,0],principal_components[:,1])
    plt.xlabel('First principal component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA via Manual Calculation')
    plt.show()

def scikit_learn_pca():
    # # REQUIRES INTERNET CONNECTION AND FIREWALL ACCESS
    # cancer_dictionary = load_breast_cancer()

    # print("keys:", cancer_dictionary.keys())

    # print("target:", cancer_dictionary['target'])

    # plt.figure(figsize=(8,6))
    # plt.scatter(principal_components[:,0],principal_components[:,1],c=cancer_dictionary['target'])
    # plt.xlabel('First principal component')
    # plt.ylabel('Second Principal Component')
    # plt.title('PCA via Manual Calculation with Target Coloring')
    # plt.show()

    ## PCA with Scikit-Learn

    df = pd.read_csv('/Users/tess/Desktop/MLE2025/ML-Masterclass/UNZIP_FOR_NOTEBOOKS_FINAL (1)/DATA/cancer_tumor_data_features.csv')
    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(df)

    pca = PCA(n_components=2)

    principal_components = pca.fit_transform(scaled_X)

    plt.figure(figsize=(8,6))
    plt.scatter(principal_components[:,0],principal_components[:,1])
    plt.xlabel('First principal component')
    plt.ylabel('Second Principal Component')
    plt.title('PCA via Scikit-Learn')
    plt.show()
    

    df_comp = pd.DataFrame(pca.components_,index=['PC1','PC2'],columns=df.columns)
    plt.figure(figsize=(20,3),dpi=150)
    sns.heatmap(df_comp,annot=True)
    plt.title("PCA Component Weights")
    plt.show()

    print("pca.explained_variance_ratio_:", pca.explained_variance_ratio_)
    print("np.sum(pca.explained_variance_ratio_):", np.sum(pca.explained_variance_ratio_))

    pca_30 = PCA(n_components=30)
    pca_30.fit(scaled_X)

    pca_30.explained_variance_ratio_

    np.sum(pca_30.explained_variance_ratio_)

    explained_variance = []

    for n in range(1,30):
        pca = PCA(n_components=n)
        pca.fit(scaled_X)
        
        explained_variance.append(np.sum(pca.explained_variance_ratio_))

    plt.plot(range(1,30),explained_variance)
    plt.xlabel("Number of Components")
    plt.ylabel("Variance Explained")
    plt.title("Elbow Plot for PCA Components vs Explained Variance")
    plt.show()

if __name__ == "__main__":
    # manual_pca()
    scikit_learn_pca()

