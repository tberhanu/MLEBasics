import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import classification_report,accuracy_score



class AdaBoost:
	def __init__(self):
		"""
		Boosting:
			- is not a ML algorithm itself, but it can be applied to any ML algorithms to combine them.
			- is the process of aggregating a bunch of weak learners(aka Stumps) and combining into an overall strong ensemble model via WEIGHTED SUM.
			- develop trees in SERIES, unlike Random Forest's building trees in parallel, which means develop one tree and use your learnings
			  from that tree and the dataset to build the next tree.
			- After each round, it adjusts the weights of the training samples to focus more on the ones that were misclassified by assigning more weight.
			- Learning Rate is not fixed, unlike Gradient Boosting, so Better Learners get Higher Learning Rate.

		Stump:
			- simplest Decision Tree (one root and 2 leaves), limited only to one feature.
		
		* AdaBoost uses an ensemble of weak learners that learn slowly in series.
		  Each subsequent 't' weak learner is built using a re-weighted data set from the 't-1' weak learner.
		  Weights of misclassified data points will be increased for the next weak learner.
		  For learner not good in predicting, alpha(x) will be low.

		Unlike Random Forest, it's possible to overfit with AdaBoost, however it takes many trees to do this, 
		and usually error has already stabilized way before enough trees are added to cause overfitting.

		"""
		df = pd.read_csv("../../../files/final_files/DATA/mushrooms.csv")
		# classifying poisonous mushrooms vs edible mushrooms

		# EDA
		sns.countplot(data=df,x='class')
		plt.show()

		stat = df.describe()
		stat2 = df.describe().transpose()

		plt.figure(figsize=(14,6),dpi=200)
		feature_unique = df.describe().transpose().reset_index().sort_values('unique')
		# sns.barplot(data=df.describe().transpose().reset_index().sort_values('unique'),x='index',y='unique')
		sns.barplot(data=feature_unique,x='index',y='unique')
		plt.xticks(rotation=90)
		plt.show()

		# Train Test Split
		X = df.drop('class',axis=1)
		missing_df = X.isnull()
		missings_per_column = missing_df.sum()
		X = pd.get_dummies(X,drop_first=True) # 0's and 1's or True/False
		y = df['class']

		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=101)

		# Modeling
		model = AdaBoostClassifier(n_estimators=1, algorithm='SAMME')
		model.fit(X_train,y_train)

		predictions = model.predict(X_test)

		classification_reports = classification_report(y_test,predictions)

		feature_importances = model.feature_importances_

		index = model.feature_importances_.argmax() # most important feature index, 22

		most_imp_feature = X.columns[index] # most important feature/column name, 'odor_n'

		sns.countplot(data=df,x='odor',hue='class')
		plt.title("most imp feature 'odor'")
		plt.show()


		## Analyzing performance as more weak learners are added.
		num_of_cols = len(X.columns)

		error_rates = []

		# for n in range(1,96): # takes a while
		for n in range(1,16): # takes a while

		    model = AdaBoostClassifier(n_estimators=n, algorithm='SAMME')
		    # Technically, we can pass 'base_estimator' to AdaBoostClassifier; otherwise it will take 
		    # DecisionTreeClassifier initialized with max_depth=1 (aka STUMP) by default.
		    model.fit(X_train,y_train)
		    preds = model.predict(X_test)
		    err = 1 - accuracy_score(y_test,preds)
		    
		    error_rates.append(err)

		plt.plot(range(1,16),error_rates)
		plt.title("elbow x=range y=error")
		plt.show()


		feature_importances2 = model.feature_importances_
		index2 = model.feature_importances_.argmax() # most important feature index, 82
		most_imp_feature2 = X.columns[index2] # most important feature/column name, 'spore-print-color_w'

		feats = pd.DataFrame(index=X.columns,data=feature_importances2,columns=['Importance']) # converting to df

		imp_feats = feats[feats['Importance']>0] # filtering df for +ve importance
		sns.barplot(data=imp_feats, x=imp_feats.index, y='Importance')
		plt.title("+ve Importance")
		plt.xticks(rotation=90)
		plt.show()

		plt.figure(figsize=(14, 6), dpi=150)
		sns.barplot(data=imp_feats.sort_values("Importance"), x=imp_feats.index, y='Importance')
		plt.title("+ve Importance Sorted")
		plt.xticks(rotation=90)
		plt.show()

		sns.countplot(data=df,x='spore-print-color',hue='class')
		plt.title("most imp feature 'spore-print-color_w'")
		plt.show()


		imp_feats_sorted = imp_feats.sort_values("Importance")

		plt.figure(figsize=(14,6),dpi=200)
		sns.barplot(data=imp_feats.sort_values('Importance'),x=imp_feats.sort_values('Importance').index,y='Importance')
		plt.xticks(rotation=90)
		plt.show()


		sns.countplot(data=df,x='habitat',hue='class')
		plt.title("countplot 'habitat' hue=class")
		plt.show()



if __name__ == "__main__":
	adaboost = AdaBoost()


