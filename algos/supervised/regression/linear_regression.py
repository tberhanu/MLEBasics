import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
import requests
from joblib import dump, load


class MyLinearRegression:

	def __init__(self, df, X=None, y=None):
		self.df = df
		self.reg_model = LinearRegression()
		self.X, self.y =  self._dataset_label_extraction("sales")

		self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

		self.final_model = LinearRegression()


	def _dataset_label_extraction(self, label):
		X = df.drop(label, axis=1)
		y = df[label]
		return X, y
	
	def train_test_split(self, test_size, random_state):
		if self.X.empty or self.y.empty:
			print("Missed either Dataset or Label.")
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)

	def train(self):
		self.reg_model.fit(self.X_train, self.y_train)

	def predict(self):
		if self.X_test.empty:
			print("No Test Data Available.")
			return
		test_predictions = self.reg_model.predict(self.X_test)
		return test_predictions

	def evaluate(self, test_predictions):
		MAE = mean_absolute_error(self.y_test,test_predictions)
		MSE = mean_squared_error(self.y_test,test_predictions)
		RMSE = np.sqrt(MSE)

	def train_final_model(self, isPlotted=False):
		# Once we are satisfied on our model performance, need to train with ALL data for production.
		self.final_model.fit(self.X, self.y)
		y_hat = self.final_model.predict(self.X)
		if isPlotted:
			fig,axes = plt.subplots(nrows=1,ncols=3,figsize=(16,6))
			axes[0].plot(df['TV'],df['sales'],'o')
			axes[0].plot(df['TV'],y_hat,'o',color='red')
			axes[0].set_ylabel("Sales")
			axes[0].set_title("TV Spend")

			axes[1].plot(df['radio'],df['sales'],'o')
			axes[1].plot(df['radio'],y_hat,'o',color='red')
			axes[1].set_title("Radio Spend")
			axes[1].set_ylabel("Sales")

			axes[2].plot(df['newspaper'],df['sales'],'o')
			axes[2].plot(df['radio'],y_hat,'o',color='red')
			axes[2].set_title("Newspaper Spend");
			axes[2].set_ylabel("Sales")
			plt.tight_layout();
			plt.show()

	def saving_final_model(self):
		# Dumping our final model
		dump(self.final_model, '../../../../files/final_files/MODELS/sales_model.joblib')

	def loading_and_predict(self):
		# Loading our persisted final model, and make the prediction
		loaded_model = load('../../../../files/final_files/MODELS/sales_model.joblib')
		campaign = [[149,22,12]]
		prediction = loaded_model.predict(campaign)



if __name__ == "__main__":
	df = pd.read_csv("../../../../files/final_files/DATA/Advertising.csv")
	myLinearRegression = MyLinearRegression(df)
	myLinearRegression.train_test_split(0.3, 101)
	myLinearRegression.train()
	test_predictions = myLinearRegression.predict()
	myLinearRegression.evaluate(test_predictions)
	myLinearRegression.train_final_model()
	myLinearRegression.saving_final_model()
	myLinearRegression.loading_and_predict()









	