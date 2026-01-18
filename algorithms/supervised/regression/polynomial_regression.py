import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error
from joblib import dump, load


class MyPolynomialRegression:
	def __init__(self):
		self.df = pd.read_csv("../../../../files/final_files/DATA/Advertising.csv")

		self.X = self.df.drop('sales',axis=1)
		self.y = self.df['sales']

		self.X_train, self.X_test = None, None
		self.y_train, self.y_test = None, None
		self.poly_features = None
		self.poly_model = None

	def get_polynomial_features(self, degree=2):
		polynomial_converter = PolynomialFeatures(degree=degree,include_bias=False)

		poly_features = polynomial_converter.fit_transform(self.X)
		row_col = poly_features.shape
		x_shape = self.X.shape

		return poly_features

	def poly_features_and_train_test_split(self, degree=2):
		self.poly_features = self.get_polynomial_features(degree)
		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.poly_features, self.y, test_size=0.3, random_state=101)

	def train_model(self):

		self.poly_model = LinearRegression(fit_intercept=True)
		self.poly_model.fit(self.X_train,self.y_train)
		

	def get_error_metric_evaluation(self):

		test_predictions = self.poly_model.predict(self.X_test)

		MAE = mean_absolute_error(self.y_test,test_predictions)
		MSE = mean_squared_error(self.y_test,test_predictions)
		RMSE = np.sqrt(MSE)

	def getting_optimal_degree(self):

		# TRAINING ERROR PER DEGREE
		train_rmse_errors = []
		# TEST ERROR PER DEGREE
		test_rmse_errors = []

		for d in range(1,10):
		    
		    # CREATE POLY DATA SET FOR DEGREE "d"
		    polynomial_converter = PolynomialFeatures(degree=d,include_bias=False)
		    poly_features = polynomial_converter.fit_transform(self.X)
		    
		    # SPLIT THIS NEW POLY DATA SET
		    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(poly_features, self.y, test_size=0.3, random_state=101)
		    
		    # TRAIN ON THIS NEW POLY SET
		    model = LinearRegression(fit_intercept=True)
		    model.fit(self.X_train,self.y_train)
		    
		    # PREDICT ON BOTH TRAIN AND TEST
		    train_pred = model.predict(self.X_train)
		    test_pred = model.predict(self.X_test)
		    
		    # Calculate Errors
		    
		    # Errors on Train Set
		    train_RMSE = np.sqrt(mean_squared_error(self.y_train,train_pred))
		    
		    # Errors on Test Set
		    test_RMSE = np.sqrt(mean_squared_error(self.y_test,test_pred))

		    # Append errors to lists for plotting later
		    
		   
		    train_rmse_errors.append(train_RMSE)
		    test_rmse_errors.append(test_RMSE)

		return train_rmse_errors, test_rmse_errors

	def plot_error_metrics(self):

		train_rmse_errors, test_rmse_errors = self.getting_optimal_degree()
		plt.plot(range(1,6),train_rmse_errors[:5],label='TRAIN')
		plt.plot(range(1,6),test_rmse_errors[:5],label='TEST')
		plt.xlabel("Polynomial Complexity (Degree)")
		plt.ylabel("RMSE")
		plt.legend()
		plt.show()

		plt.plot(range(1,10),train_rmse_errors,label='TRAIN')
		plt.plot(range(1,10),test_rmse_errors,label='TEST')
		plt.xlabel("Polynomial Complexity (Degree)")
		plt.ylabel("RMSE")
		plt.legend()
		plt.show()


		plt.plot(range(1,10),train_rmse_errors,label='TRAIN')
		plt.plot(range(1,10),test_rmse_errors,label='TEST')
		plt.xlabel("Polynomial Complexity (Degree)")
		plt.ylabel("RMSE")
		plt.ylim(0,100)
		plt.legend()
		plt.show()


	def deploy_final_model(self):
		final_poly_converter = PolynomialFeatures(degree=3,include_bias=False)

		final_model = LinearRegression()
		final_model.fit(final_poly_converter.fit_transform(self.X), self.y)

		dump(final_model, 'sales_poly_model.joblib') 
		dump(final_poly_converter,'poly_converter.joblib')

	def loading_deployed_model(self):
		loaded_poly = load('poly_converter.joblib')
		loaded_model = load('sales_poly_model.joblib')
		campaign = [[149,22,12]]
		campaign_poly = loaded_poly.transform(campaign)
		prediction = loaded_model.predict(campaign_poly)




if __name__ == "__main__":

	myPolynomialRegression = MyPolynomialRegression()
	myPolynomialRegression.get_polynomial_features(2)
	myPolynomialRegression.poly_features_and_train_test_split(2)
	myPolynomialRegression.train_model(2)
	myPolynomialRegression.get_error_metric_evaluation()
	myPolynomialRegression.getting_optimal_degree()
	myPolynomialRegression.plot_error_metrics()
	myPolynomialRegression.deploy_final_model()
	myPolynomialRegression.loading_deployed_model()

	

	
	

	



