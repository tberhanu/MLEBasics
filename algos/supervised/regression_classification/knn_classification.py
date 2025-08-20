import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix,classification_report
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve, RocCurveDisplay
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

class KNNClassificatioin:
	def __init__(self):
		self.df = pd.read_csv('./UNZIP_FOR_NOTEBOOKS_FINAL/DATA/gene_expression.csv')
		# print("dyptes: ", self.df.dtypes) # gives datatype of each column
		# print("type: ", type(self.df)) # <class 'pandas.core.frame.DataFrame'>

		self.X = self.df.drop('Cancer Present',axis=1)
		self.y = self.df['Cancer Present']

		self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42)
		scaler = StandardScaler()

		self.scaled_X_train = scaler.fit_transform(self.X_train)
		self.scaled_X_test = scaler.transform(self.X_test)

	def visualize_data(self):
		sns.scatterplot(x='Gene One',y='Gene Two',hue='Cancer Present',data=self.df)
		plt.title("scatterplot without transparency alpha")
		plt.show()

		sns.scatterplot(x='Gene One',y='Gene Two',hue='Cancer Present',data=self.df,alpha=0.7) #'alpha' to add transparency for dots on top of each other
		plt.title("scatterplot with transparency alpha")
		plt.show()

		sns.scatterplot(x='Gene One',y='Gene Two',hue='Cancer Present',data=self.df,alpha=0.7, style='Cancer Present') #'style' shape to see better on top of each other
		plt.title("scatterplot with 'style' shape")
		plt.show()

		sns.scatterplot(x='Gene One',y='Gene Two',hue='Cancer Present',data=self.df,alpha=0.7)
		plt.xlim(2,6)
		plt.ylim(3,10)
		plt.title("Zooming In at (X, Y) range")
		plt.show()

		sns.pairplot(data=self.df, hue='Cancer Present')
		plt.title("pairplot")
		plt.show()

	def preliminary_model(self):

		knn_model = KNeighborsClassifier(n_neighbors=1)
		knn_model.fit(self.scaled_X_train,self.y_train)
		# Understanding KNN and Choosing K Value
		full_test = pd.concat([self.X_test,self.y_test],axis=1)

		# Quick Visualization
		plt.figure(figsize=(10,6))
		sns.scatterplot(x='Gene One', y='Gene Two', hue='Cancer Present', data=full_test,alpha=0.7)
		# plt.legend(bbox_to_anchor=(1.11, 0.5), title="Cancer Present")
		plt.legend(loc=(1.0, 0.5), title="Cancer Present")
		plt.show()

		## Model Evaluation
		y_pred = knn_model.predict(self.scaled_X_test)
		accuracyScore = accuracy_score(self.y_test,y_pred)
		confusionMatrix = confusion_matrix(self.y_test,y_pred)
		classificationReport = classification_report(self.y_test,y_pred) 
		# print(classificationReport) # for better visualization

		# Let's see Elbow Plot for optimal k value
		self.knn_elbow_method_optimization(k_start=1, k_end=30)

	def knn_elbow_method_optimization(self, k_start, k_end):
		## Elbow Method for Choosing Reasonable K Values

		# **NOTE: This uses the test set for the hyperparameter selection of K.**

		test_error_rates = []


		for k in range(k_start, k_end):
		    knn_model = KNeighborsClassifier(n_neighbors=k)
		    knn_model.fit(self.scaled_X_train,self.y_train) 
		   
		    y_pred_test = knn_model.predict(self.scaled_X_test)
		    
		    test_error = 1 - accuracy_score(self.y_test,y_pred_test)
		    test_error_rates.append(test_error)


		plt.figure(figsize=(10,6),dpi=200)
		plt.plot(range(k_start, k_end),test_error_rates,label='Test Error')
		plt.legend()
		plt.ylabel('Error Rate')
		plt.xlabel("K Value")
		plt.show()

	def gridsearch_cv_pipeline(self, k_start, k_end, cv, scoring):
		## Full Cross Validation Grid Search for K Value
		### Creating a Pipeline to find K value

		scaler = StandardScaler()
		knn = KNeighborsClassifier()
		param_keys = knn.get_params().keys() # keys for our 'param_grid' down

		# Highly recommend string code matches variable name!
		operations = [('scaler',scaler),('knn',knn)]

		pipe = Pipeline(operations)

		# k_values = list(range(1,20))
		k_values = list(range(k_start, k_end))
		total_runs = len(k_values)

		param_grid = {'knn__n_neighbors': k_values} # attention to the naming convention
		# chosen_string_name + two underscores + param_key from 'param_keys' => 'knn__n_neighbors'

		full_cv_classifier = GridSearchCV(pipe, param_grid, cv=cv, scoring=scoring) # cv for K-fold
		# if cv=5, dataset is divided into 5 equal-sized folds or subsets, so model training and evaluation process reapeated 5 times
		# and in each iteration 4-folds used for Training Set and 1-fold for Evaluation Set.
		# taking 5 runs averaged for less bias and mitigate the risk of overfitting

		# Use full X and y if you DON'T want a hold-out test set
		# Use self.X_train and self.y_train if you DO want a holdout test set (self.X_test,self.y_test)
		full_cv_classifier.fit(self.X_train,self.y_train) # NOTE: not using scaled_X_train as PIPELINE will do it for us !!!

		optimal_pipeline_params = full_cv_classifier.best_estimator_.get_params() # for both 'scaler' and 'knn' in pipeline

		cv_results_keys = full_cv_classifier.cv_results_.keys()
		cv_results_keys_df = pd.DataFrame(full_cv_classifier.cv_results_)
		each_run_test_score = full_cv_classifier.cv_results_['mean_test_score']

		# Pandas df plotting
		cv_results_keys_df['mean_test_score'].plot() # Accuracy Plot for all K's, not loss plot
		plt.title("pandas df Accuracy plot")
		plt.xlabel("k values")
		plt.ylabel("mean test score")
		plt.show()

		# the formal way to plot the above
		plt.plot(k_values, full_cv_classifier.cv_results_['mean_test_score'])
		plt.title("matplotlib Accuracy plot")
		plt.xlabel("k values")
		plt.ylabel("mean test score")
		plt.show()


		return full_cv_classifier

	def final_knn_model(self):
		"""
		Final Model¶
		We just saw that our GridSearch recommends a K=14 (in line with our alternative Elbow Method). 
		Let's now use the PipeLine again, but this time, no need to do a grid search, instead we will evaluate on our hold-out Test Set.

		"""
		k_start, k_end, cv, scoring = 1, 30, 5, 'accuracy'
		full_cv_classifier = self.gridsearch_cv_pipeline(k_start, k_end, cv, scoring)
		optimal_pipeline_params = full_cv_classifier.best_estimator_.get_params() # for both 'scaler' and 'knn' in pipeline

		n_neighbors = optimal_pipeline_params['knn__n_neighbors']

		scaler = StandardScaler()
		knn = KNeighborsClassifier(n_neighbors=n_neighbors)
		operations = [('scaler',scaler),('knn',knn)]

		pipe = Pipeline(operations)
		pipe.fit(self.X_train,self.y_train)

		full_predictions = pipe.predict(self.X_test)

		classificationReport = classification_report(self.y_test, full_predictions)

		# Let's predict for single sample
		single_sample = self.X_test.iloc[40]

		# pipe_predictions2 = pipe.predict(single_sample.values.reshape(1, -1)) # works fine
		single_sample_df = pd.DataFrame([single_sample])
		single_prediction = pipe.predict(single_sample_df)

		# pipe_probabilities = pipe.predict_proba(single_sample.values.reshape(1, -1)) # works fine
		single_probability = pipe.predict_proba(single_sample_df)

		# Another single prediction for our new patient
		new_patient = [[3.8,  6.4]] # not single list, but double
		single_prediction2 = pipe.predict(new_patient)
		single_probability2 = pipe.predict_proba(new_patient)

		print("done.")

	





if __name__ == "__main__":
	knn = KNNClassificatioin()
	# knn.visualize_data()
	# knn.preliminary_model()
	knn.final_knn_model()















