import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D 


class Tools:
	def __init__(self):
		self.df = pd.read_csv("/Users/tess/Desktop/MLE2025/projects/UNZIP_FOR_NOTEBOOKS_FINAL/DATA/hearing_test.csv")

	def exploratory_data_analysis(self):

		print("df info: \n", self.df.info())
		print("df head: \n", self.df.head())
		print("(rows, cols)=: ", self.df.shape)
		print("df descriptions: \n", self.df.describe())

	def countplot(self, feature, plt_title):
		print(f"'{feature}'' value distribution: ", self.df[feature].value_counts())
		sns.countplot(data=self.df,x=feature)
		plt.title(plt_title)
		plt.show()

	def boxplot(self, feature1, feature2, plt_title):
		sns.boxplot(x=feature1,y=feature2,data=self.df)
		plt.title(plt_title)
		plt.show()

	def scatterplot(self, feature1, feature2, plt_title, hue_feature3=None):
		sns.scatterplot(x=feature1,y=feature2,data=self.df,hue=hue_feature3)
		plt.title(plt_title)
		plt.show()

	def pairplot(self, plt_title, hue_feature=None):
		sns.pairplot(self.df,hue=hue_feature)
		plt.title(plt_title)
		plt.show()

	def heatmap_corr(self, plt_title):
		sns.heatmap(self.df.corr(),annot=True)
		plt.title(plt_title)
		plt.show()

	def scatter_3d(self):
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		ax.scatter(self.df['age'], self.df['physical_score'], self.df['test_result'],c=self.df['test_result']) # coloring based on 'test_result'
		# ax.scatter(self.df['age'], self.df['physical_score'], self.df['test_result'],c='r') # just coloring RED


		ax.set_xlabel('age') # Set the x-axis label
		ax.set_ylabel('physical_score') # Set the y-axis label
		ax.set_zlabel('test_result') # Set the z-axis label
		plt.title('3D Scatter Plot Example') # Add a title to the plot
		
		plt.show()

	def scatter_plot_3d_4d(self):

		"""
		If only 3d plot needed, just use c=fourth_variable as a COLOR by using one of the three Variables or just directly like for RED >> c='r'
		Also, you may not need to add a colorbar, 'cbar'.
		"""
		# Generate some example data
		n_points = 100
		x = np.random.rand(n_points) * 10
		y = np.random.rand(n_points) * 10
		z = np.random.rand(n_points) * 10
		fourth_variable = np.random.rand(n_points) * 50  # This will determine the color
		# print("fourth_variable: ", fourth_variable)

		# Create a figure and a 3D subplot
		fig = plt.figure(figsize=(10, 8))
		ax = fig.add_subplot(111, projection='3d')

		# Create the 3D scatter plot, coloring by the fourth variable
		scatter_plot = ax.scatter(x, y, z, c=fourth_variable, cmap='viridis', s=50) # 'viridis' is a good default colormap

		# Set labels for the axes
		ax.set_xlabel('X-axis')
		ax.set_ylabel('Y-axis')
		ax.set_zlabel('Z-axis')

		# Add a colorbar
		cbar = fig.colorbar(scatter_plot, ax=ax, pad=0.1) # Add padding to avoid overlapping with plot
		cbar.set_label('Fourth Variable Value')

		plt.title('3D Scatter Plot with Color Representing a Fourth Variable')
		plt.show()


if __name__ == "__main__":
	tools = Tools()
	tools.exploratory_data_analysis()

	feature, plt_title = "test_result", "1. sns.countplot"
	tools.countplot(feature, plt_title)

	feature1, feature2, plt_title = "test_result", "age", "2. sns.boxplot"
	tools.boxplot(feature1, feature2, plt_title)

	feature1, feature2, plt_title = "test_result", "physical_score", "3. sns.boxplot"
	tools.boxplot(feature1,feature2, plt_title)

	feature1, feature2, plt_title, hue_feature3 = "age", "physical_score", "4. sns.scatterplot", "test_result"
	tools.scatterplot(feature1, feature2, plt_title, hue_feature3)

	feature1, feature2, plt_title = "physical_score", "test_result","5. sns.scatterplot"
	tools.scatterplot(feature1,feature2, plt_title)

	feature1, feature2, plt_title = "age", "test_result", "6. sns.scatterplot"
	tools.scatterplot(feature1, feature2, plt_title)

	plt_title, hue_feature = "7. sns.pairplot", "test_result"
	tools.pairplot(plt_title, hue_feature)

	plt_title = "8. sns.heatmap corr"
	tools.heatmap_corr(plt_title)

	tools.scatter_3d()
	tools.scatter_plot_3d_4d()
















