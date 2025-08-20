import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report,accuracy_score


class GradientBoosting:
	def __init__(self):
		"""
		*** Tree based models like DecisionTreeClassifier, RandomForest, and GradientBoosting are OK if strings converted to numbers via
            OrdinalEncoder or OneHotEncoder, but the modeliteslef doesn't need the features to be scaled or normalized. 

		Similar to AdaBoost, weak learners are created in series in order to produce a strong ensemble model, 
		but it makes use of the RESIDUAL ERROR (y - y_hat) for learning, not WEIGHTED DATA POINTS.
		Unlike AdaBoost, larger trees allowed in Gradient Boosting, not just Stump.
		Unlike AdaBoost, learning rate coefficient same for all weak learners.
		Lower learning rate mean more trees needed as each subsequent tree has little "say"

		In Gradient Boosting, like Neural Network, we minimize a loss function — but instead of backpropagation and weight updates using gradients,
	    we add new models (usually decision trees) that predict the negative gradient (i.e., the residual error).

	    Decision trees are non-parametric (they don't assume any fixe form for the data) and great at capturing complex patterns, and Gradient Boosting turns them into a powerful 
	    ensemble by stacking them in a way that each tree learns from the mistakes of the previous ones.

		"""

	def basics(self):
		df = pd.read_csv("../../../files/final_files/DATA/mushrooms.csv")


		## Data Prep
		X = df.drop('class',axis=1)
		y = df['class']
		X = pd.get_dummies(X,drop_first=True) # Both GradientBoostingClassifier and GradientBoostingRegressor doesn’t 
											  # handle non-numeric data directly, so you must encode categorical features.

		## Train Test Split 
		X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=101)

		## Gradient Boosting and Grid Search with CV
		# help(GradientBoostingClassifier)

		# param_grid = {"n_estimators":[1,5,10,20,40,100],'learning_rate': [0.1,0.5,0.2],'max_depth':[3,4,5,6]}
		param_grid = {"n_estimators":[1,5,10,20,40,100],'max_depth':[3,4,5,6]}

		gb_model = GradientBoostingClassifier() # GradientBoostingRegressor() for Regression tasks
		# implicitly, DecisionTreeClassifier or DecisionTreeRegressor is used, unless you eant ot customize it.
		grid = GridSearchCV(gb_model,param_grid)


		### Fit to Training Data with CV Search
		grid.fit(X_train,y_train) # may take some time depending your computer and the param_grid
		bestParams = grid.best_params_


		## Performance
		predictions = grid.predict(X_test)

		classification_reports = classification_report(y_test,predictions)

		# featureImportances = grid.best_estimator_.feature_importances_

		feat_import = grid.best_estimator_.feature_importances_ # this gives importance for more features than the orginal features bc of One-HotEncoder
		

		imp_feats_df = pd.DataFrame(index=X.columns,data=feat_import,columns=['Importance'])

		imp_feats_df_sorted = imp_feats_df.sort_values("Importance",ascending=False)

		stat = imp_feats_df.describe().transpose()

		imp_feats_df_filtered = imp_feats_df[imp_feats_df['Importance'] > 0.000527]

		imp_feats_df_filtered_sorted = imp_feats_df_filtered.sort_values('Importance')

		plt.figure(figsize=(14,6),dpi=100)
		sns.barplot(data=imp_feats_df.sort_values('Importance'),x=imp_feats_df.sort_values('Importance').index,y='Importance')
		plt.xticks(rotation=90)
		# plt.title("barplot sorted importance")
		plt.show()

		#########################################################################
		# Let's figure out the importance with respect to the original features:
		# Create a DataFrame to inspect
		feat_import_df = pd.DataFrame({
		    'feature': X.columns,
		    'importance': feat_import
		})
		# Group by original feature prefix
		feat_import_df['base_feature'] = feat_import_df['feature'].str.split('_').str[0] # extracting 'base_feature'
		agg_importance = feat_import_df.groupby('base_feature')['importance'].sum().sort_values(ascending=False)

		# Convert Series to DataFrame for plotting
		agg_importance_df = agg_importance.reset_index()

		plt.figure(figsize=(14,6),dpi=100)
		sns.barplot(data=agg_importance_df, x='base_feature', y='importance')		
		plt.xticks(rotation=90)
		plt.title("imp of original feature")
		plt.show()
		#########################################################################




if __name__ == "__main__":
	gradient_boosting = GradientBoosting()
	gradient_boosting.basics()




