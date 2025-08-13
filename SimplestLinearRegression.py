import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error
import requests
from joblib import dump, load

class SimpleOperations:

	def __init__(self, df):
		self.df = df

	def display_data_head(self):
		print(self.df.head)

	def add_total_spend_feature(self):
		self.df['total_spend'] = self.df['TV'] + self.df['radio'] + self.df['newspaper']

	def visualize_corr_scatterplot(self, feature1='total_spend', feature2='sales'):
		sns.scatterplot(x=feature1, y=feature2, data=self.df)
		plt.show()

	def visualize_corr_regplot(self, feature1='total_spend', feature2='sales'):
		sns.regplot(x=feature1, y=feature2, data=self.df)
		plt.show()

class SimplestLinearRegression:
	"""
		Here, we use np.polyfit() to train and np.polyval() to predict, only for Single Feature, 1D.
		So, if you try to have your training dataset, X, with 3 features as below, will get ERROR.
			X = df[['TV','radio','newspaper']]
			y = df['sales']
			np.polyfit(X,y,1) # 'TypeError: expected 1D vector for x'

		Later, we will use Scikit-Learn Regression Model for Multiple Features.


	"""
	def __init__(self, total_spend, sales):
		self.X = total_spend 
		self.y = sales
		self.coefficients = None
		self.sample_data = np.linspace(0,500,100)

	def get_coefficients(self, degree=1):
		self.coefficients = np.polyfit(self.X, self.y, degree) # training the model
		return self.coefficients

	def predict_sales(self, total_spend, degree):
		coefficients = self.get_coefficients(degree)
		# predicted_sales =  0.04868788*potential_spend + 4.24302822  manually calculating via coefficients
		predicted_sales = np.polyval(coefficients, total_spend) # trained model predicting
		return predicted_sales


	def plotting(self, sample_data, predicted_sales):
		
		sns.scatterplot(x='total_spend',y='sales',data=df)
		plt.plot(self.sample_data, predicted_sales,color='red')
		plt.show()

	def plot_feature_to_label(self, df):
		fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16,6))
		# Plotting each feature versus the Sales
		axes[0].plot(df['TV'], df['sales'],'o')
		axes[0].set_ylabel("Sales")
		axes[0].set_title("TV Spend")

		axes[1].plot(df['radio'], df['sales'],'o')
		axes[1].set_title("Radio Spend")
		axes[1].set_ylabel("Sales")

		axes[2].plot(df['newspaper'], df['sales'],'o')
		axes[2].set_title("Newspaper Spend");
		axes[2].set_ylabel("Sales")
		plt.tight_layout();
		plt.show()


	def plot_feature_cross_relationships(self, df):
		# Relationships between features
		sns.pairplot(df,diag_kind='kde')
		plt.show()


	
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
		# Once we are satisfied on our model performance, need to train with ALL data for production
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
		dump(self.final_model, './UNZIP_FOR_NOTEBOOKS_FINAL/DATA/sales_model.joblib')

	def loading_and_predict(self):
		# Loading our persisted final model, and make the prediction
		loaded_model = load('./UNZIP_FOR_NOTEBOOKS_FINAL/DATA/sales_model.joblib')
		campaign = [[149,22,12]]
		loaded_model.predict(campaign)



	def others(self):
		quartet = pd.read_csv('anscombes_quartet1.csv')
		# y = 3.00 + 0.500x
		quartet['pred_y'] = 3 + 0.5 * quartet['x']
		quartet['residual'] = quartet['y'] - quartet['pred_y']

		sns.scatterplot(data=quartet,x='x',y='y')
		sns.lineplot(data=quartet,x='x',y='pred_y',color='red')
		plt.vlines(quartet['x'],quartet['y'],quartet['y']-quartet['residual'])

		sns.kdeplot(quartet['residual'])

		sns.scatterplot(data=quartet,x='y',y='residual')
		plt.axhline(y=0, color='r', linestyle='--')


if __name__ == "__main__":
	df = pd.read_csv("./UNZIP_FOR_NOTEBOOKS_FINAL/DATA/Advertising.csv")
	############# Just Simple Operations ##############
	op = SimpleOperations(df)
	op.display_data_head()
	op.add_total_spend_feature()
	op.visualize_corr_scatterplot()
	op.visualize_corr_regplot()
	
	########### Simplest, 1D (Single Feature), Numpy Linear Regression ################
	model = SimplestLinearRegression(df['total_spend'], df['sales'])
	model.predict_sales(200)
	predicted_sales = model.predict_sales(model.sample_data, 1)
	model.plotting(self.sample_data, predicted_sales)

 	############# Still 1D (Single Feature) But More DEGREE, Polynomial #################
	predicted_sales = model.predict_sales(model.sample_data, 3)
	model.plotting(model.sample_data, predicted_sales)
	model.plot_feature_to_label(df)
	model.plot_feature_cross_relationships(df)




	
