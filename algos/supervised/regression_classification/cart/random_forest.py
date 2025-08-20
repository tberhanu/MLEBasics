import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor





class RandomForest:
    def __init__(self):
        """
        "Random Forest does not overfit. You can run as many trees as you want. It is fast." (Leo Breiman's Official Page @ www.stat.berkeley.edu)

        Why Random Forest, instead of keep using Decision Trees ?
            1. In Decision Tree, neither Ginin Impurity nor Informaton Gain(Entropy) guarantee usage of ALL features as both
               selects features GREEDILY.
            2. In Decision Tree, ROOT FEATURE NODE will always be the same for the same set of rules like same max depth, same # of splits etc. so the
               ROOT FEATURE NODE has huge influence over the tree all the way down to the leaves which leads some of the featues may NEVER be used
               that may also lead to overfitting.

        For Random Forest:
            1. Randmly pick subsets of features, with replacement.
            2. Use Information Gain(Entropy) or Gini Impurity to decide which one of the randomly picked features to be used as a splitting criteria.
            3. Do the same for all the nodes down to the leaf, i.e. randomly different feature node used for splitting
            4. At last, report either the MAJORITY VOTE for Classification (together with the PROBABILITY) or the AVERAGE VALUE for Regression.

        Scikit-learn hyperparameters for Decision Tree and Random Forest are similar mostly, except the belows added for Random Forest:
            1. n_estimators=100     >> How many total number of decision trees to use in the forest ?
                                    >> "the more feature, the more trees" is the basic principle, but to quikcly decide:
                                        >> 1. Start with the default 100 trees, and GridSearch for higher values, and take the best performer
                                        >> 2. Plot 'Error' vs 'Number of Estimators(Trees)', similar to Elbow Method
                                    >> What if too many trees ? 
                                        >> Trees become highly correlated and no more revealing any more new information and may end up
                                           getting duplicated trees, but this won't trigger overfitting !!!
            2. max_features='sqrt'  >> How many features to include in each randomly picked subset ?(randomly picked with replacement)
                                    >> Start with the default 'sqrt', and then GridSearch of other possible values like N/3.
            3. bootstrap=True       >> Should we allow for bootstrap sampling of each training subset of features ?
                                    >> bootstrap ~> randomly pick subsets of rows of data, with replacement
                                    >> Building the Tree based on randomly selected subset of rows, instead of all the data rows.
                                    Why Bootstarpping ?
                                        >> adding another parameter to reduce correlation between Trees as we are training Trees not only
                                           on different subset of feature columns but also on different data rows. This means some of the
                                           rows used in the training, and the rest, Out Of Bag Samples, are not used for training.
            4. oob_score=False      >> Should we calculate OOB(Out Of Bag) error during the training ?
                                    >> It's all about using our Out Of Bag Samples (rows not used for training) for testing a single Tree
                                       as a Testing Dataset as an optional way of measuring performance.
                                    >> Setting True/False, doesn't really affect the training process.


        
        Note: For Tree Based methods like a Single Decision Tree or a Random Forest, no need to worry about having to scale or
              standardize or normalize any of the feature data points since we're just going to be doing a split based on a
              single feature.


        


        """
        pass

    def classification(self):
        df = pd.read_csv("../../../../files/final_files/DATA/penguins_size.csv")

        df = df.dropna() # dropping all the missing data points

        X = pd.get_dummies(df.drop('species',axis=1),drop_first=True)
        y = df['species']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

        # help(RandomForestClassifier)

        # Use 10 random trees
        model = RandomForestClassifier(n_estimators=10,max_features='sqrt',random_state=101)

        model.fit(X_train,y_train)

        preds = model.predict(X_test)

        ## Evaluation
        confusionMatrix = confusion_matrix(y_test,preds)

        # plot_confusion_matrix(model,X_test,y_test)
        display1 = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test) # without Normalizing
        display1.plot()
        plt.title("ConfusionMatrixDisplay RawData")
        plt.show()

        classificationReport = classification_report(y_test, preds)

        # featureImportances = model.feature_importances_

        # Good thing about Decision Tree is it's very interpretable by humans
        feature_importances = model.feature_importances_ # relative importance of features in the decision making
        # feature importances order align with the column/feature of our df, but we can still make another df to clearly display
        feature_importances_df = pd.DataFrame(index=X.columns,data=feature_importances,columns=['Feature Importance'])
        feature_importances_df_sorted = feature_importances_df.sort_values('Feature Importance', ascending=False)
        # You may see that some of the feature's has Zero importance, it's b/c the algo may not consider them to split the tree
        # unless you really force it to do so which may result in overfitting.

        # import pdb;pdb.set_trace()

        ## Choosing correct number of trees
        # Let's explore if continually adding more trees improves performance...


        test_error = []

        for n in range(1,40):
            # Use n random trees
            model = RandomForestClassifier(n_estimators=n,max_features='sqrt')
            model.fit(X_train,y_train)
            test_preds = model.predict(X_test)
            test_error.append(1-accuracy_score(test_preds,y_test))


        plt.plot(range(1,40),test_error,label='Test Error')
        plt.xlabel("n_estimators")
        plt.ylabel("errors")
        plt.legend()
        plt.show()


    def classification_gridsearch(self):
        # counterfit or true bill

        # Random Forest - HyperParameter Exploration
        df = pd.read_csv("../../../../files/final_files/DATA/data_banknote_authentication.csv")

        # Replace inf and -inf with NaN
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df = df.dropna() # dropping all the missing data points


        sns.pairplot(df,hue='Class')
        plt.show()

        X = df.drop("Class",axis=1)
        y = df["Class"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=101)
        # x_test and y_test will be our HOLD-OUT test since we're gonna perform a GridSearch.
        # GridSearch will automatically handles splitting training data into 'training' and 'validation folds'(5 by default) for evaluation, and
        # we don't need to explicity provide a separate validation set unless we want to customize the process.
        # grid = GridSearchCV(RandomForestClassifier(), param_grid, cv=22) # customizing it for cv=22, 22 fold.

        n_estimators=[64,100,128,200]
        max_features= [2,3,4]
        bootstrap = [True,False]
        # oob_score = [True,False] # Out of bag estimation only available if bootstrap=Tru
        oob_score = [False]


        param_grid = {'n_estimators':n_estimators,
             'max_features':max_features,
             'bootstrap':bootstrap,
             'oob_score':oob_score}  # Note, oob_score only makes sense when bootstrap=True! So, expect warning.

        rfc = RandomForestClassifier()
        grid = GridSearchCV(rfc,param_grid)
        # grid = GridSearchCV(rfc,param_grid, error_score='raise')

        try:
            grid.fit(X_train,y_train)
        except Exception as e:
            print(e)

        bestParams = grid.best_params_
        try:
            predictions = grid.predict(X_test)
        except Exception as e:
            print(e)

        # print(classification_report(y_test,predictions))
        classificationReport = classification_report(y_test, predictions)

        confusionMatrix = confusion_matrix(y_test,predictions)

        # plot_confusion_matrix(grid,X_test,y_test)
        display1 = ConfusionMatrixDisplay.from_estimator(grid, X_test, y_test) # without Normalizing
        display1.plot()
        plt.title("ConfusionMatrixDisplay RawData")
        plt.show()

        # No underscore, reports back original oob_score parameter
        original_oob_score = grid.best_estimator_.oob_score

        # # With underscore, reports back fitted attribute of oob_score
        # fitted_attr_oob_score = grid.best_estimator_.oob_score_


        ## Understanding Number of Estimators (Trees)
        # Let's plot out error vs. Number of Estimators

        errors = []
        misclassifications = []

        for n in range(1,64): # takes time
            rfc = RandomForestClassifier(n_estimators=n,bootstrap=True,max_features= 2)
            rfc.fit(X_train,y_train)
            preds = rfc.predict(X_test)
            err = 1 - accuracy_score(preds,y_test)
            n_missed = np.sum(preds != y_test) # (True=1) if matched; else (False=0)
            errors.append(err)
            misclassifications.append(n_missed)

        plt.plot(range(1,64),errors)
        plt.xlabel("n_estimators")
        plt.ylabel("errors")
        plt.show()

        plt.plot(range(1,64),misclassifications)
        plt.xlabel("n_estimators")
        plt.ylabel("misclassifications")
        plt.show()


    def optimal_regression_model(self): 
        """
        Trying different regression models upto random forest regression:
            1. Linear Regression 
            2. Polynomial Regression (PIPELINE)
            3. KNN Regression 
            4. Decision Tree Regression
            5. Support Vector Regression 
            6. Random Forest Regression
            Boosted Trees Regression
                7. Gradient Boosting
                8. AdaBoosting
    
               


        """
        
        df = pd.read_csv("../../../../files/final_files/DATA/rock_density_xray.csv")
        df.columns=['Signal',"Density"] # renaming columns with convenient name

        plt.figure(figsize=(12,8),dpi=200)
        sns.scatterplot(x='Signal',y='Density',data=df)
        plt.show() # curved pattern tells us Linear Regression is not the right algorithm to fit

        X = df['Signal'].values.reshape(-1,1) # reshape to convert 1D to 2D
        y = df['Density']
        """
        Note: Reshape your data as below:
            1. array.reshape(-1, 1) if your data is a single feature.
            2. array.reshape(1, -1) if it contains a single sample (single row)
        """

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

        # 1. Linear Regression
        lr_model = LinearRegression()
        lr_model.fit(X_train,y_train)
        lr_preds = lr_model.predict(X_test)

        # Even if MAE and RMSE looks good on average, TRICKY.
        MAE = mean_absolute_error(y_test, lr_preds)
        RMSE = np.sqrt(mean_squared_error(y_test,lr_preds))

        # Let's create arteficial signal_range, and plot the prediction to see how it fit.
        signal_range = np.arange(0,100)
        lr_output = lr_model.predict(signal_range.reshape(-1,1))

        plt.figure(figsize=(12,8))
        sns.scatterplot(x='Signal',y='Density',data=df,color='black')
        plt.plot(signal_range,lr_output)
        plt.show() # the horizontal line of prediction confirm linear regression being unfit


        # 2. Polynomial Regression
        ## Attempting with a Polynomial Regression Model
        # Let's explore why our standard regression approach of a polynomial could be difficult to fit here, 
        # keep in mind, we're in a fortunate situation where we can easily visualize results of y vs x.

        model = LinearRegression()
        title = "linear regresion"
        self.run_model(model,X_train,y_train,X_test,y_test, df, title)

        pipe = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
        title = "polynomial degree=2 pipeline"
        self.run_model(pipe,X_train,y_train,X_test,y_test, df, title)

        ## Comparing Various Polynomial Orders

        pipe = make_pipeline(PolynomialFeatures(10),LinearRegression())
        title = "polynomial degree=10 pipeline"
        self.run_model(pipe,X_train,y_train,X_test,y_test, df, title)



        ## 3. KNN Regression
        preds = {}
        k_values = [1,5,10]
        for n in k_values:
            
            model = KNeighborsRegressor(n_neighbors=n)
            title = f"kNeighborsRegressor n_neighbors={n}"
            self.run_model(model,X_train,y_train,X_test,y_test, df, title)


        ## 4. Decision Tree Regression
        model = DecisionTreeRegressor()
        title = "DecisionTreeRegressor"
        self.run_model(model,X_train,y_train,X_test,y_test, df, title)

        leaves_count = model.get_n_leaves()


        ## 5. Support Vector Regression
        param_grid = {'C':[0.01,0.1,1,5,10,100,1000],'gamma':['auto','scale']}
        svr = SVR()

        grid = GridSearchCV(svr,param_grid)
        title = "SupportVectorRegression GridSearchCV"
        self.run_model(grid,X_train,y_train,X_test,y_test, df, title)

        bestEstimator = grid.best_estimator_


        ## 6. Random Forest Regression
        # help(RandomForestRegressor)
        trees = [10,50,100]
        for n in trees:
            
            model = RandomForestRegressor(n_estimators=n)
            title = f"RandomForestRegressor n_estimators={n}"
            self.run_model(model,X_train,y_train,X_test,y_test, df, title)




        ## 7. Gradient Boosting
        # help(GradientBoostingRegressor)
        model = GradientBoostingRegressor()
        title = "Gradient Boosting Regressor"
        self.run_model(model,X_train,y_train,X_test,y_test, df, title)



        ## 8. Adaboost
        model = AdaBoostRegressor()
        title = "Adaboost Regressor"
        self.run_model(model,X_train,y_train,X_test,y_test, df, title)



    def run_model(self, model, X_train, y_train, X_test, y_test, df, title):
    
        # Fit Model Training
        model.fit(X_train,y_train)
        
        # Get Metrics
        preds = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test,preds))
        mae = mean_absolute_error(y_test,preds)
        print(f'RMSE for {title}: {rmse}')
        print(f'MAE for {title}: {mae}')
        
        # Plot results
        signal_range = np.arange(0,100)
        signal_preds = model.predict(signal_range.reshape(-1,1))
        
        
        plt.figure(figsize=(12,6),dpi=150)
        sns.scatterplot(x='Signal',y='Density',data=df,color='black')
        plt.plot(signal_range,signal_preds)
        plt.title(title)
        plt.show()

if __name__ == "__main__":

    random_forest = RandomForest()
    random_forest.classification()
    random_forest.classification_gridsearch()
    random_forest.optimal_regression_model()







