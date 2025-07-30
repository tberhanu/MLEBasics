import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class CategoricalData:
	def __init__(self):
		self.df = pd.read_csv("/Users/tess/Desktop/MLE2025/projects/UNZIP_FOR_NOTEBOOKS_FINAL/DATA/ALTERED/Ames_NO_Missing_Data.csv")
		#print(self.df.info())

	def convert_to_string(self, feature_name):
		print(f"======dtypes of '{feature_name}' before converting to str: \n", self.df.dtypes[feature_name])
		self.df[feature_name] = self.df[feature_name].apply(str)
		print(f"======dtypes of '{feature_name}' after converting to str: \n", self.df.dtypes[feature_name])

	def create_dummy_variables(self):
		df_nums = self.df.select_dtypes(exclude='object')
		print("===================================NUMBERS: \n")
		print(df_nums.info())
		df_objs = self.df.select_dtypes(include='object')
		print("===================================OBJECTS/STRINGS: \n")
		print(df_objs.info())

		# creating dummy variables for Object/String Data Types
		df_objs = pd.get_dummies(df_objs,drop_first=True)

		final_df = pd.concat([df_nums,df_objs],axis=1)
		print("===================================Final DF: \n")
		print(final_df)
		print("===================================Final DF Shape: ", final_df.shape)

		return final_df

	def correlation_analysis(self):
		"""

		Final Thoughts
		Keep in mind, we don't know if 274 columns is very useful. More columns doesn't necessarily lead to better results. 
		In fact, we may want to further remove columns (or later on use a model with regularization to choose important columns for us). 
		What we have done here has greatly expanded the ratio of rows to columns, which may actually lead to worse performance.
		(however you don't know until you've actually compared multiple models/approaches).

		"""

		final_df = self.create_dummy_variables()
		print("Correlation Values with 'SalePrice': \n", final_df.corr()['SalePrice'].sort_values())
		print("Clearly 'Overall Qual' has significant impact on the 'SalePrice'")

		print("Saving the final_df: ")
		final_df.to_csv('/Users/tess/Desktop/MLE2025/projects/UNZIP_FOR_NOTEBOOKS_FINAL/DATA/ALTERED/AMES_Final_DF.csv')



if __name__ == "__main__":
	categoricalData = CategoricalData()
	categoricalData.convert_to_string('MS SubClass')
	categoricalData.create_dummy_variables()
	categoricalData.correlation_analysis()


