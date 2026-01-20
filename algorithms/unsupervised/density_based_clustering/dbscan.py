
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN


class DensityBasedSpatialClusteringOfApplicationsWithNoise:
    def __init__(self):
        # self.eps = eps
        # self.min_samples = min_samples
        # self.model = DBSCAN(eps=self.eps,min_samples=self.min_samples)
        self.two_blobs = pd.read_csv('/Users/tess/Desktop/MLE2025/ML-Masterclass/UNZIP_FOR_NOTEBOOKS_FINAL (1)/DATA/cluster_two_blobs.csv')
        self.two_blobs_outliers = pd.read_csv('/Users/tess/Desktop/MLE2025/ML-Masterclass/UNZIP_FOR_NOTEBOOKS_FINAL (1)/DATA/cluster_two_blobs_outliers.csv')
    
    def display_categories(self, model, data, title):
        labels = model.fit_predict(data)
        sns.scatterplot(data=data,x='X1',y='X2',hue=labels,palette='Set1')
        plt.title(title)
        plt.show()

    def visualize_data(self):
        dbscan = DBSCAN()
        self.display_categories(dbscan, self.two_blobs, title="Default Epsilon and Min Samples")
        self.display_categories(dbscan, self.two_blobs_outliers, title="Default Epsilon and Min Samples")

        # Tiny Epsilon --> Tiny Max Distance --> Everything is an outlier (class=-1)
        dbscan = DBSCAN(eps=0.001)
        self.display_categories(dbscan, self.two_blobs_outliers, title="Tiny Epsilon")


        # Huge Epsilon --> Huge Max Distance --> Everything is in the same cluster (class=0)
        dbscan = DBSCAN(eps=10)
        self.display_categories(dbscan, self.two_blobs_outliers, title="Huge Epsilon")


        # How to find a good epsilon?
        plt.figure(figsize=(10,6),dpi=200)
        dbscan = DBSCAN(eps=1)
        self.display_categories(dbscan, self.two_blobs_outliers, title="eps=1")

        print("dbscan.labels_")
        print(dbscan.labels_)
        print("dbscan.labels_ == -1")
        print(dbscan.labels_ == -1)
        print("np.sum(dbscan.labels_ == -1)")
        print(np.sum(dbscan.labels_ == -1))
        print("100 * np.sum(dbscan.labels_ == -1) / len(dbscan.labels_)")
        print(100 * np.sum(dbscan.labels_ == -1) / len(dbscan.labels_))


    def epsilon_run(self):
        outlier_percent = []
        number_of_outliers = []

        for eps in np.linspace(0.001,10,100):
            
            # Create Model
            dbscan = DBSCAN(eps=eps)
            dbscan.fit(self.two_blobs_outliers)
            
            # Log Number of Outliers
            number_of_outliers.append(np.sum(dbscan.labels_ == -1))
            
            # Log percentage of points that are outliers
            perc_outliers = 100 * np.sum(dbscan.labels_ == -1) / len(dbscan.labels_)
            
            outlier_percent.append(perc_outliers)
            

        sns.lineplot(x=np.linspace(0.001,10,100),y=outlier_percent)
        plt.ylabel("Percentage of Points Classified as Outliers")
        plt.xlabel("Epsilon Value")
        plt.show()


        sns.lineplot(x=np.linspace(0.001,10,100),y=number_of_outliers)
        plt.ylabel("Number of Points Classified as Outliers")
        plt.xlabel("Epsilon Value")
        plt.xlim(0,1)
        plt.show()

        sns.lineplot(x=np.linspace(0.001,10,100),y=outlier_percent)
        plt.ylabel("Percentage of Points Classified as Outliers")
        plt.xlabel("Epsilon Value")
        plt.ylim(0,5)
        plt.xlim(0,2)
        plt.hlines(y=1,xmin=0,xmax=2,colors='red',ls='--')
        plt.show()

        # How to find a good epsilon?
        dbscan = DBSCAN(eps=0.4)
        self.display_categories(dbscan,self.two_blobs_outliers, title="Epsilon=0.4")

        sns.lineplot(x=np.linspace(0.001,10,100),y=number_of_outliers)
        plt.ylabel("Number of Points Classified as Outliers")
        plt.xlabel("Epsilon Value")
        plt.ylim(0,10)
        plt.xlim(0,6)
        plt.hlines(y=3,xmin=0,xmax=10,colors='red',ls='--')
        plt.show()

        # How to find a good epsilon?
        dbscan = DBSCAN(eps=0.75)
        self.display_categories(dbscan,self.two_blobs_outliers, title="Epsilon=0.75")
    
    def min_samples_run(self):
        outlier_percent = []

        for n in np.arange(1,100):
            
            # Create Model
            dbscan = DBSCAN(min_samples=n)
            dbscan.fit(self.two_blobs_outliers)
            
            # Log percentage of points that are outliers
            perc_outliers = 100 * np.sum(dbscan.labels_ == -1) / len(dbscan.labels_)
            
            outlier_percent.append(perc_outliers)


        sns.lineplot(x=np.arange(1,100),y=outlier_percent)
        plt.ylabel("Percentage of Points Classified as Outliers")
        plt.xlabel("Minimum Number of Samples")
        plt.show()

        num_dim = self.two_blobs_outliers.shape[1]

        dbscan = DBSCAN(min_samples=2*num_dim)
        self.display_categories(dbscan,self.two_blobs_outliers, title="Min Samples = 2 * Num Dimensions")

        num_dim = self.two_blobs_outliers.shape[1]

        dbscan = DBSCAN(eps=0.75,min_samples=2*num_dim)
        self.display_categories(dbscan,self.two_blobs_outliers, title="Epsilon=0.75 and Min Samples = 2 * Num Dimensions")

        dbscan = DBSCAN(min_samples=1)
        self.display_categories(dbscan,self.two_blobs_outliers, title="Min Samples = 1")

        dbscan = DBSCAN(eps=0.75,min_samples=1)
        self.display_categories(dbscan, self.two_blobs_outliers, title="Epsilon=0.75 and Min Samples = 1")


if __name__ == "__main__":
    
    # eps=0.5,min_samples=5
    dbscan = DensityBasedSpatialClusteringOfApplicationsWithNoise()
    # dbscan.visualize_data()
    dbscan.epsilon_run()
    dbscan.min_samples_run()

    
   