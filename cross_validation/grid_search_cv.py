import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


class GridSearch:
	def __init__(self):
		self.df = pd.read_csv("../../../../files/final_files/DATA/Advertising.csv")
		self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

	def formatting_data(self):
		## CREATE X and y
		X = self.df.drop('sales',axis=1)
		y = self.df['sales']

		# TRAIN TEST SPLIT
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.3, random_state=101)

		# SCALE DATA
		scaler = StandardScaler()
		scaler.fit(self.X_train)
		self.X_train = scaler.transform(self.X_train)
		self.X_test = scaler.transform(self.X_test)


	def make_model_via_gridsearchcv(self):
		base_elastic_model = ElasticNet()

		param_grid = {'alpha':[0.1,1,5,10,50,100], 'l1_ratio':[.1, .5, .7, .9, .95, .99, 1]}
		# verbose number a personal preference
		grid_model = GridSearchCV(estimator=base_elastic_model,
		                          param_grid=param_grid,
		                          scoring='neg_mean_squared_error',
		                          cv=5,
		                          verbose=2)

		grid_model.fit(self.X_train, self.y_train) # Training on all available CV
		optimal_model = grid_model.best_estimator_
		optimal_params = grid_model.best_params_
		
		return grid_model

	def predict(self, grid_model):
		y_pred = grid_model.predict(self.X_test)
		print("Predictions: \n", y_pred)
		print("MSE: ", mean_squared_error(self.y_test, y_pred))


if __name__ == "__main__":
	gridSearch = GridSearch()
	gridSearch.formatting_data()
	grid_model = gridSearch.make_model_via_gridsearchcv()
	gridSearch.predict(grid_model)

