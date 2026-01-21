import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster import hierarchy
"""
Note: For distance, we will use Euclidean Distance, sqrt(a_diff^2 + b_diff^2 + c_diff^2 + ... + n_diff^2 ).
      If we use MinMaxScaler, then all values between 0 and 1, so the max distance
      between two points is sqrt(N), where N = number of features.
## Agglomerative Clustering Approach
0. Load Data and Preprocess (Handle Missing Values, Categorical Variables via One-Hot Encoding, Scaling via MinMaxScaler)
1. Start by visualizing the data with heatmaps and clustermaps
2. Get some ideas on number of clusters with visualizations
3. Create agglomerative clustering model with chosen number of clusters
4. Get cluster labels by fitting and predicting on scaled data via the model
5. Visualize the clusters with scatterplots using hue as cluster labels (hue=cluster_labels)

            ## Exploring Number of Clusters with Dendrograms
1. Create agglomerative clustering model with n_clusters=None and distance_threshold=0
Note: distance_threshold: is the linkage distance threshold above which clusters will not be merged.
      So, if distance_threshold = 0 >> no clusters will be merged.
          if distance_threshold = sqrt(num_samples) >> All merged in one cluster
2. Get the linkage matrix from the model's children_ attribute using hierarchy.linkage()
3. Plot the dendrogram using hierarchy.dendrogram()
4. Optionally, truncate the dendrogram for better visualization using truncate_mode and p parameters
5. Analyze the dendrogram to decide on an appropriate number of clusters
"""
df = pd.read_csv('/Users/tess/Desktop/MLE2025/ML-Masterclass/UNZIP_FOR_NOTEBOOKS_FINAL (1)/DATA/cluster_mpg.csv')

df = df.dropna()
# df.head()
# df.describe()
df['origin'].value_counts() # 1: America, 2: Europe, 3: Asia

df_w_dummies = pd.get_dummies(df.drop('name',axis=1))
scaler = MinMaxScaler() # Scale between 0 and 1
scaled_data = scaler.fit_transform(df_w_dummies)

scaled_df = pd.DataFrame(scaled_data,columns=df_w_dummies.columns) # Convert back to DataFrame to visualize

plt.figure(figsize=(15,8))
sns.heatmap(scaled_df,cmap='magma')
plt.title('Heatmap of Scaled Data')
plt.show()
sns.clustermap(scaled_df,row_cluster=False)
plt.title('Clustermap without Row Clustering')
plt.show()
sns.clustermap(scaled_df,col_cluster=False)
plt.title('Clustermap without Column Clustering')
plt.show()

model = AgglomerativeClustering(n_clusters=4)
cluster_labels = model.fit_predict(scaled_df)

plt.figure(figsize=(12,4),dpi=200)
sns.scatterplot(data=df,x='mpg',y='weight',hue=cluster_labels)
plt.title('Agglomerative Clustering with 4 Clusters')
plt.show()
## Exploring Number of Clusters with Dendrograms
#### Assuming every point starts as its own cluster

model = AgglomerativeClustering(n_clusters=None,distance_threshold=0)
cluster_labels = model.fit_predict(scaled_df)

## Linkage Model
linkage_matrix = hierarchy.linkage(model.children_)

plt.figure(figsize=(20,10))
# Warning! This plot will take awhile!!
dn = hierarchy.dendrogram(linkage_matrix)
plt.title('Full Dendrogram')
plt.show()

plt.figure(figsize=(20,10))
dn = hierarchy.dendrogram(linkage_matrix,truncate_mode='lastp',p=48)
plt.title('Truncated Dendrogram with All Points')
plt.show()

scaled_df.describe()
scaled_df['mpg'].idxmax()
scaled_df['mpg'].idxmin()

# https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
a = scaled_df.iloc[320]
b = scaled_df.iloc[28]
dist = np.linalg.norm(a-b) # Euclidean distance between two points
print("Euclidean distance between point 320 and 28:", dist)

print("max distance between two points:", np.sqrt(len(scaled_df.columns))) # max distance between two points in scaled data

### Creating a Model Based on Distance Threshold
model = AgglomerativeClustering(n_clusters=None,distance_threshold=2)
cluster_labels = model.fit_predict(scaled_data)
np.unique(cluster_labels)

### Linkage Matrix
linkage_matrix = hierarchy.linkage(model.children_)

plt.figure(figsize=(20,10))
dn = hierarchy.dendrogram(linkage_matrix,truncate_mode='lastp',p=11)
plt.title('Truncated Dendrogram with Distance Threshold = 2')
plt.show()
