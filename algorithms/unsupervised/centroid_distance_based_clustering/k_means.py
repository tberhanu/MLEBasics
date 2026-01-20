"""
K-Means Clustering
Let's work through an example of unsupervised learning - clustering customer data.

Goal:
When working with unsupervised learning methods, its usually important to lay out a general goal. 
In our case, let's attempt to find reasonable clusters of customers for marketing segmentation and study. 
What we end up doing with those clusters would depend heavily on the domain itself, in this case, marketing.

The Data
LINK: https://archive.ics.uci.edu/ml/datasets/bank+marketing

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

class KMeansAlgorithm:
    def __init__(self):
        df = pd.read_csv("/Users/tess/Desktop/MLE2025/ML-Masterclass/UNZIP_FOR_NOTEBOOKS_FINAL (1)/DATA/bank-full.csv")
        # import pdb; pdb.set_trace()
        # df.head()
        # df.columns
        # df.info()
        self.X = pd.get_dummies(df) # Convert categorical variables to dummy/indicator variables.
        scaler = StandardScaler()
        self.scaled_X = scaler.fit_transform(self.X)

    def run(self):
        

        model = KMeans(n_clusters=2)

        cluster_labels = model.fit_predict(self.scaled_X)
        # IMPORTANT NOTE: YOUR 0s and 1s may be opposite of ours,
        # makes sense, the number values are not significant!
        # cluster_labels
        # len(self.scaled_X)
        # len(cluster_labels)
        self.X['Cluster'] = cluster_labels
        sns.heatmap(self.X.corr())

        self.X.corr()['Cluster']
        plt.figure(figsize=(12,6),dpi=200)
        self.X.corr()['Cluster'].iloc[:-1].sort_values().plot(kind='bar')
        plt.show()


    def elbow_method(self, scaled_X):
        ## Choosing K Value
        ssd = []

        for k in range(2,10):
            
            model = KMeans(n_clusters=k)
            
            
            model.fit(scaled_X)
            
            #Sum of squared distances of samples to their closest cluster center.
            ssd.append(model.inertia_)


        plt.plot(range(2,10),ssd,'o--')
        plt.xlabel("K Value")
        plt.ylabel(" Sum of Squared Distances")
        plt.show()



        # ssd
        # Change in SSD from previous K value! (SSD = Sum of Squared Distances)
        pd.Series(ssd).diff()
        pd.Series(ssd).diff().plot(kind='bar')
        plt.show()

    def color_quantization(self, n_clusters):
        """
        K‑Means Color Quantization is a technique that reduces the number of colors in an image while trying
        to keep the image looking as close to the original as possible. It’s a clever mix of math and art.

        Here’s the idea in a clean, intuitive way.

        🎨 What problem is it solving?
        A normal image might have thousands or millions of unique colors.
        K‑Means Color Quantization tries to shrink that down to something like:
        8 colors, or 16 colors, or 32 colors while still keeping the picture recognizable.

        Why do this?
            To compress images
            To create artistic posterized effects
            To simplify images for machine learning
            To reduce file size
            To extract dominant colors (like for palettes)


        Think of every pixel as a point in RGB space: [R,G,B]
        K‑Means tries to group all these pixels into K clusters.

        Each cluster represents a dominant color.

        Then:

        Every pixel in the image is replaced by the center color of its cluster

        The image now uses only K colors instead of thousands

        K‑Means might group them into clusters like:

        Cluster 1 → Blue-ish

        Cluster 2 → Yellow-ish

        Cluster 3 → Red-ish

        …

        Example effect (conceptually)
            Original image:

                1,000,000 pixels

                120,000 unique colors

            After K‑Means with K=8:

                Still 1,000,000 pixels

                Only 8 unique colors remain

                Image looks flatter, more “poster-like”


        Summary: K‑Means Color Quantization:

            Reduces the number of colors in an image
            Groups similar colors together
            Replaces them with representative colors
            Keeps the image recognizable
            Useful for compression, stylization, and color analysis

        """



        image_as_array = mpimg.imread('/Users/tess/Desktop/MLE2025/ML-Masterclass/UNZIP_FOR_NOTEBOOKS_FINAL (1)/DATA/palm_trees.jpg')
        plt.figure(figsize=(6,6),dpi=200)
        plt.imshow(image_as_array)
        plt.title("Original Image")
        plt.show()

        print("image shape:", image_as_array.shape)
        # (h,w,3 color channels)

        # Convert from 3d to 2d
        # Kmeans work for N Dimensions, but the input has to be 2D
        # You could even use more dimensions:
            # RGB -> 3D
            # RGB + brightness → 4D
            # RGB + x + y pixel location → 5D

            # Lab color space → 3D but nonlinear
        # Kmeans is designed to train on 2D input data: (number of samples, number of features) 
        # so we can reshape the above strip by using (h,w,c) ---> (h * w,c)
        (h,w,c) = image_as_array.shape
        image_as_array2d = image_as_array.reshape(h*w,c) #(number of samples, number of features)

        model = KMeans(n_clusters=n_clusters)
        labels = model.fit_predict(image_as_array2d)
        print("labels shape:", labels.shape) 
        print("labels:", labels) # expected to have 6 unique values: 0,1,2,3,4,5
        # THESE ARE THE 6 RGB COLOR CODES!
        print("model.cluster_centers_:", model.cluster_centers_)
        rgb_codes = model.cluster_centers_.round(0).astype(int)
        print("rgb_codes:", rgb_codes)
        quantized_image = np.reshape(rgb_codes[labels], (h, w, c))
        print("Image as array 3D shape: ", image_as_array.shape)
        print("Original Image Array 2D shape:", image_as_array2d.shape)
        print("quantized_image shape:", quantized_image.shape)
        plt.figure(figsize=(6,6),dpi=200)
        plt.imshow(quantized_image)
        plt.title(f"Quantized Image with {n_clusters} colors")
        plt.show()

if __name__ == "__main__":
    kmeans = KMeansAlgorithm()
    # kmeans.run()
    # kmeans.elbow_method(kmeans.scaled_X)
    kmeans.color_quantization(n_clusters=6)