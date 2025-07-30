
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

		print("============About to Scale Data using StandardScaler.")
		scaler = StandardScaler()
		scaler.fit(self.X_train) # calculates Mean and Standard Deviation of each feature (only in training data)
		self.X_train = scaler.transform(self.X_train)
		self.X_test = scaler.transform(self.X_test)

	def ridge_regression_model(self, alpha=10):
		print("============Ridge Regression Model Construction.")
		ridge_model = Ridge(alpha=10)
		ridge_model.fit(self.X_train, self.y_train)

		test_predictions = ridge_model.predict(self.X_test)
		print("Test Predictions: ", test_predictions)

		MAE = mean_absolute_error(self.y_test, test_predictions)
		MSE = mean_squared_error(self.y_test, test_predictions)
		RMSE = np.sqrt(MSE)
		print(f"Tested on Test Data: MAE={MAE}, MSE={MSE}, RMSE={RMSE}")

		train_predictions = ridge_model.predict(self.X_train)
		MAE = mean_absolute_error(self.y_train, train_predictions)
		print(f"Tested on All Dataset: MAE={MAE}")
		print("-------------------------------------------------------")


	def ridgeCV_model(self):
		print("============RidgeCV Model Construction.")
		ridge_cv_model = RidgeCV(alphas=(0.1, 1.0, 10.0),scoring='neg_mean_absolute_error')
		ridge_cv_model.fit(self.X_train, self.y_train)
		print(f"RidgeCV Model Optimal Alpha: {ridge_cv_model.alpha_}")
		test_predictions = ridge_cv_model.predict(self.X_test)

		MAE = mean_absolute_error(self.y_test, test_predictions)
		MSE = mean_squared_error(self.y_test, test_predictions)
		RMSE = np.sqrt(MSE)
		print(f"MAE={MAE}, MSE={MSE}, RMSE={RMSE}")
		print("-------------------------------------------------------")

		train_predictions = ridge_cv_model.predict(self.X_train)
		MAE = mean_absolute_error(self.y_train, train_predictions)
		print(f"Tested on All Dataset: MAE={MAE}")
		print(f"Ridge CV Model Coefficients: {ridge_cv_model.coef_}")

	def lassoCV_model(self):
		print("============LassoCV Model Construction.")
		lasso_cv_model = LassoCV(eps=0.1,n_alphas=100,cv=5)
		lasso_cv_model.fit(self.X_train, self.y_train)
		print(f"Lasso CV Model Optimal Alpha: {lasso_cv_model.alpha_}")

		test_predictions = lasso_cv_model.predict(self.X_test)
		MAE = mean_absolute_error(self.y_test, test_predictions)
		MSE = mean_squared_error(self.y_test, test_predictions)
		RMSE = np.sqrt(MSE)
		print(f"Tested on Test Data: MAE={MAE}, MSE={MSE}, RMSE={RMSE}")
		print("-------------------------------------------------------")


		train_predictions = lasso_cv_model.predict(self.X_train)
		MAE = mean_absolute_error(self.y_train, train_predictions)
		print(f"Tested on All Dataset: MAE={MAE}")

	def elasticnet_model(self):
		print("============Elastic Net Model Construction.")
		elasticnet_model = ElasticNetCV(l1_ratio=[.1, .5, .7,.9, .95, .99, 1],tol=0.01)
		elasticnet_model.fit(self.X_train, self.y_train)
		print(f"Elastic Net Model Optimal L1 Ratio: {elasticnet_model.l1_ratio_}")

		test_predictions = elasticnet_model.predict(self.X_test)
		MAE = mean_absolute_error(self.y_test, test_predictions)
		MSE = mean_squared_error(self.y_test, test_predictions)
		RMSE = np.sqrt(MSE)
		print(f"MAE={MAE}, MSE={MSE}, RMSE={RMSE}")

		train_predictions = elasticnet_model.predict(self.X_train)
		MAE = mean_absolute_error(self.y_train, train_predictions)
		print(f"Tested on All Dataset: MAE={MAE}")
		print(f"Elastic Net Model Coefficients: {elasticnet_model.coef_}")
		print("-------------------------------------------------------")


if __name__ == "__main__":
	myRegularization = myRegularization()
	print("DF Head=========: ", myRegularization.df.head())
	myRegularization.scaling_data()
	myRegularization.ridge_regression_model()
	myRegularization.ridgeCV_model()
	myRegularization.lassoCV_model()
	myRegularization.elasticnet_model()

