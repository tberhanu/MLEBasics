
import numpy as np
from PolynomialRegression import MyPolynomialRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import ElasticNetCV

class myRegularization(MyPolynomialRegression):
	def __init__(self):
		super().__init__()


	def scaling_data(self):
		self.poly_features_and_train_test_split(3)
		scaler = StandardScaler()
		scaler.fit(self.X_train) # calculates Mean and Standard Deviation of each feature (only in training data)
		self.X_train = scaler.transform(self.X_train)
		self.X_test = scaler.transform(self.X_test)

	def ridge_regression_model(self, alpha=10):
		ridge_model = Ridge(alpha=10)
		ridge_model.fit(self.X_train, self.y_train)

		test_predictions = ridge_model.predict(self.X_test)

		MAE = mean_absolute_error(self.y_test, test_predictions)
		MSE = mean_squared_error(self.y_test, test_predictions)
		RMSE = np.sqrt(MSE)

		train_predictions = ridge_model.predict(self.X_train)
		MAE = mean_absolute_error(self.y_train, train_predictions)


	def ridgeCV_model(self):
		ridge_cv_model = RidgeCV(alphas=(0.1, 1.0, 10.0),scoring='neg_mean_absolute_error')
		ridge_cv_model.fit(self.X_train, self.y_train)
		ridge_optimal_alpha = ridge_cv_model.alpha_
		test_predictions = ridge_cv_model.predict(self.X_test)

		MAE = mean_absolute_error(self.y_test, test_predictions)
		MSE = mean_squared_error(self.y_test, test_predictions)
		RMSE = np.sqrt(MSE)

		train_predictions = ridge_cv_model.predict(self.X_train)
		MAE = mean_absolute_error(self.y_train, train_predictions)
		coefficients = ridge_cv_model.coef_

	def lassoCV_model(self):
		lasso_cv_model = LassoCV(eps=0.1,n_alphas=100,cv=5)
		lasso_cv_model.fit(self.X_train, self.y_train)
		lasso_optimal_alpha = lasso_cv_model.alpha_

		test_predictions = lasso_cv_model.predict(self.X_test)
		MAE = mean_absolute_error(self.y_test, test_predictions)
		MSE = mean_squared_error(self.y_test, test_predictions)
		RMSE = np.sqrt(MSE)

		train_predictions = lasso_cv_model.predict(self.X_train)
		MAE = mean_absolute_error(self.y_train, train_predictions)

	def elasticnet_model(self):
		elasticnet_model = ElasticNetCV(l1_ratio=[.1, .5, .7,.9, .95, .99, 1],tol=0.01)
		elasticnet_model.fit(self.X_train, self.y_train)
		elasticnet_optimal_l1_ratio = elasticnet_model.l1_ratio_

		test_predictions = elasticnet_model.predict(self.X_test)
		MAE = mean_absolute_error(self.y_test, test_predictions)
		MSE = mean_squared_error(self.y_test, test_predictions)
		RMSE = np.sqrt(MSE)

		train_predictions = elasticnet_model.predict(self.X_train)
		MAE = mean_absolute_error(self.y_train, train_predictions)
		elasticnet_coef = elasticnet_model.coef_


if __name__ == "__main__":
	myRegularization = myRegularization()
	print("DF Head=========: ", myRegularization.df.head())
	myRegularization.scaling_data()
	myRegularization.ridge_regression_model()
	myRegularization.ridgeCV_model()
	myRegularization.lassoCV_model()
	myRegularization.elasticnet_model()

