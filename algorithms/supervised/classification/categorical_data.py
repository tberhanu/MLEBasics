import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class CategoricalData:
	def __init__(self):
		self.df = pd.read_csv("../../../../files/final_files/DATA/ALTERED/Ames_NO_Missing_Data.csv")

	def convert_to_string(self, feature_name):
		type1 = self.df.dtypes[feature_name]
		self.df[feature_name] = self.df[feature_name].apply(str)
		type2 = self.df.dtypes[feature_name]

	def create_dummy_variables(self):
		df_nums = self.df.select_dtypes(exclude='object')
		df_objs = self.df.select_dtypes(include='object')

		# creating dummy variables for Object/String Data Types
		df_objs = pd.get_dummies(df_objs,drop_first=True)
		final_df = pd.concat([df_nums,df_objs],axis=1)

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
		correlation = final_df.corr()['SalePrice'].sort_values() # Correlation Values with 'SalePrice
		final_df.to_csv('../../../../files/final_files/DATA/ALTERED/AMES_Final_DF.csv')



if __name__ == "__main__":
	categoricalData = CategoricalData()
	categoricalData.convert_to_string('MS SubClass')
	categoricalData.create_dummy_variables()
	categoricalData.correlation_analysis()


