
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns





class Outliers:
	def __init__(self):
		pass

	def generate_data(self):
		data = self.create_ages(50, 13, 100, 42)
		# print("Generated Data: ", data)
		return data

	# Choose a mean,standard deviation, and number of samples
	def create_ages(self, mu=50,sigma=13,num_samples=100,seed=42):

	    # Set a random seed in the same cell as the random call to get the same values as us
	    # We set seed to 42 (42 is an arbitrary choice from Hitchhiker's Guide to the Galaxy)
	    np.random.seed(seed)

	    sample_ages = np.random.normal(loc=mu,scale=sigma,size=num_samples)
	    sample_ages = np.round(sample_ages,decimals=0)
	    
	    return sample_ages

	def visualize_distribution_and_outliers(self, sample_data):
		series = pd.Series(sample_data)
		print("Some Statistics: ", series.describe())

		d = sns.displot(sample_data, bins=10, kde=False)
		d.fig.suptitle("sns.displot(sample_data, bins=10, kde=False)")
		plt.show()


		sns.boxplot(sample_data)
		plt.title("sns.boxplot(sample_data)")
		plt.show()

	def ames_dataset(self):
		df = pd.read_csv("/Users/tess/Desktop/MLE2025/projects/UNZIP_FOR_NOTEBOOKS_FINAL/DATA/Ames_Housing_Data.csv")
		print(df.head())
		print("Shape: ", df.shape)
		print("2D relationship between features: ")
		
		# sns.heatmap(df.corr())
		# plt.show()

		# features_corr_with_SalePrice = df.corr()['SalePrice'].sort_values()
		# print("features_corr_with_SalePrice: ", features_corr_with_SalePrice)
		print("Plotting 1")
		d1 = sns.displot(df["SalePrice"]) # 1
		d1.fig.suptitle("sns.distplot(df['SalePrice'])")
		plt.show()

		print("Plotting 2: Before Dropping Outlier Rows")
		d2 = sns.scatterplot(x='Overall Qual',y='SalePrice',data=df) # 2
		d2.set_title("sns.scatterplot(x='Overall Qual',y='SalePrice',data=df)")
		plt.show()

		print("Plotting 3: Before Dropping Outlier Rows")
		d3 = sns.scatterplot(x='Gr Liv Area',y='SalePrice',data=df) # 3
		d3.set_title("sns.scatterplot(x='Gr Liv Area',y='SalePrice',data=df)")
		plt.show()

		############### Retrieving Outlier Rows ###############################
		print("Potential Outliers: ", df[(df['Gr Liv Area']>4000) & (df['SalePrice']<400000)])
		ind_drop = df[(df['Gr Liv Area']>4000) & (df['SalePrice']<400000)].index
		print("Potential Outliers Indices which may need to be dropped")
		############### Dropping Outlier Rows ###############################
		df = df.drop(ind_drop,axis=0)
		print("df after dropping potential outlier indices: ", df)

		print("Plotting 4: After Dropping Outlier Rows")
		ax = sns.scatterplot(x='Gr Liv Area',y='SalePrice',data=df) # 4 
		ax.set_title("sns.scatterplot(x='Gr Liv Area',y='SalePrice',data=df)")
		plt.show()

		print("Plotting 5: After Dropping Outlier Rows")
		ax2 = sns.scatterplot(x='Overall Qual',y='SalePrice',data=df) # 5
		ax2.set_title("sns.scatterplot(x='Overall Qual',y='SalePrice',data=df")
		plt.show()

		df.to_csv("/Users/tess/Desktop/MLE2025/projects/UNZIP_FOR_NOTEBOOKS_FINAL/DATA/ALTERED/Ames_outliers_removed.csv",index=False)
		print("After dropping potential outliers, data saved to ALTERED folder.")



if __name__ == "__main__":
	outliers = Outliers()
	# outliers.generate_data()
	# sample_data = outliers.create_ages(mu=50,sigma=13,num_samples=100,seed=42)
	# outliers.visualize_distribution_and_outliers(sample_data)
	outliers.ames_dataset()
