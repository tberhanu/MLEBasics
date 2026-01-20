import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from svm_margin_plot import plot_svm_boundary, plot_svm_boundary2, plot_svm_boundary_with_pca
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix,classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR,LinearSVR
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay


class SupportVectorMachine:
    def __init__(self):
        """
        When it comes to Support Vectors, both for Classification and Regression, we should be both SCALING our data and
        perform a Cross Validation and Grid Search for the best parameters as it's really hard to have an intuition for
        many different data sets over reasonabel values for C, Epsilon, gamma etc.
        Since it's really large parameter search work, we need to set good amount of time to leave the computer to do the hard work for us.

        """
        pass

    def classification_SVC(self):
        df = pd.read_csv("./UNZIP_FOR_NOTEBOOKS_FINAL/DATA/mouse_viral_study.csv")
        columns = df.columns

        # Analysis
        sns.scatterplot(x='Med_1_mL',y='Med_2_mL',hue='Virus Present',data=df,palette='seismic')
        plt.show()

        # Manual Classifier Linear Line
        sns.scatterplot(x='Med_1_mL',y='Med_2_mL',hue='Virus Present',palette='seismic',data=df)
        # We want to somehow automatically create a separating hyperplane ( a line in 2D)
        x, m, b = np.linspace(0,10,100), -1, 11
        y = m*x + b
        plt.plot(x,y,'k')
        plt.show()


        ### SVC (we skip Train Test Split and other stuffs)

        y = df['Virus Present']
        X = df.drop('Virus Present',axis=1) 

        ## Hyper Parameters
        ### C, Regularization parameter: The strength of the regularization is inversely proportional to C. 
        # Must be strictly positive. The penalty is a squared l2 penalty.
        # C => tells us how many points you are allowing to be within the soft margin
        # if Bigger C => allowing Fewer points within the soft margin, less regularization
        # if Smaller C => allowing More points within the soft margin, more regularization

        # Let's explore variety of models to be the optimal one
        model = SVC(kernel='linear', C=1000) 
        model.fit(X, y)
        # linear kernel so won't transform anything, and hyperplane will be straight line. 
        # very high C may result with less/no points within the soft margin
        plot_svm_boundary(model,X,y, "kernel=linear C=1000") # will help to understand the effects we can change with SVM

        # plot_svm_boundary2("original scikit code") # original Scikit Developer's code
        # plot_svm_boundary_with_pca(model, X, y, "2D: kernel=linear C=1000") # custom code based on scikit code for more features
        

        model = SVC(kernel='linear', C=0.05)
        model.fit(X, y)
        # still linear kernel so hyperplane is still straight line, but very small C may result in many points within the soft margin
        # keeping C lower and lower, makes the margin softer and softer.
        plot_svm_boundary(model,X,y, "kernel=linear C=0.05")

        # Generally, hard to have some kind of intuition about what the appropriate value of C is esp for more than 2 features
        # so better to run some sort of cross validation network search to get the best/optimal C. 

        # Kernel
        # Choosing a Kernel
        # rbf - Radial Basis Function, the default kernel
        model = SVC(kernel='rbf', C=1)
        # Here, kernel will take our original dataset and will project it onto a higher dimensional space.
        model.fit(X, y)
        plot_svm_boundary(model,X,y, "kernel=rbf C=1")    

        model = SVC(kernel='sigmoid')
        model.fit(X, y)
        plot_svm_boundary(model,X,y, "kernel=sigmoid C=1")

        #### Degree (poly kernels only)
        model = SVC(kernel='poly', C=1,degree=1)
        model.fit(X, y)
        plot_svm_boundary(model,X,y, "kernel=poly C=1 degree=1")

        model = SVC(kernel='poly', C=1,degree=2)
        model.fit(X, y)
        plot_svm_boundary(model,X,y, "kernel=poly C=1 degree=2")

        # gamma
        # gamma : {'scale', 'auto'} or float, default='scale' Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.

        # - if ``gamma='scale'`` (default) is passed then it uses
        #   1 / (n_features * X.var()) as value of gamma,
        # - if 'auto', uses 1 / n_features.

        model = SVC(kernel='rbf', C=1,gamma='scale')
        model.fit(X, y)
        plot_svm_boundary(model,X,y, "kernel=rbf C=1 gamma=scale")

        model = SVC(kernel='rbf', C=1,gamma='auto')
        model.fit(X, y)
        plot_svm_boundary(model,X,y, "kernel=rbf C=1 gamma=auto")
        model = SVC(kernel='rbf', C=1,gamma=0.01)
        model.fit(X, y)
        plot_svm_boundary(model,X,y, "kernel=rbf C=1 gamma=0.01")

        model = SVC(kernel='rbf', C=1,gamma=2)
        model.fit(X, y)
        plot_svm_boundary(model,X,y, "kernel=rbf C=1 gamma=2")

        # Grid Searching
        svm = SVC()
        param_grid = {'C':[0.01,0.1,1],'kernel':['linear','rbf']}
        grid_model = GridSearchCV(svm,param_grid)

        # Note again we didn't split Train|Test
        grid_model.fit(X,y)

        # 100% accuracy (as expected)
        bestScore = grid_model.best_score_

        bestParams = grid_model.best_params_

        return grid_model


    def regression_SVR(self):
        df = pd.read_csv('./UNZIP_FOR_NOTEBOOKS_FINAL/DATA/cement_slump.csv')
        columns = df.columns

        plt.figure(figsize=(8, 8), dpi=100)
        sns.heatmap(df.corr(), annot=True)
        plt.show()

        corr_with_strength = df.corr()['Compressive Strength (28-day)(Mpa)']

        sns.heatmap(df.corr(),cmap='viridis')
        plt.show()

        X = df.drop('Compressive Strength (28-day)(Mpa)',axis=1)
        y = df['Compressive Strength (28-day)(Mpa)']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
        scaler = StandardScaler()
        scaled_X_train = scaler.fit_transform(X_train)
        scaled_X_test = scaler.transform(X_test)


        """
        Support Vector Machines - Regression
        There are three different implementations of Support Vector Regression: SVR, NuSVR and LinearSVR. 
        LinearSVR provides a faster implementation than SVR but only considers the linear kernel, while NuSVR implements a slightly different formulation than SVR and LinearSVR.

        Setting C: C is 1 by default and it’s a reasonable default choice. If you have a lot of noisy observations you should decrease it: decreasing C corresponds to more regularization.

        LinearSVC and LinearSVR are less sensitive to C when it becomes large, and prediction results stop improving after a certain threshold. 
        Meanwhile, larger C values will take more time to train, sometimes up to 10 times longer

        """

        base_model = SVR()
        base_model.fit(scaled_X_train,y_train)

        base_preds = base_model.predict(scaled_X_test)

        MAE = mean_absolute_error(y_test,base_preds)
        RMSE = np.sqrt(mean_squared_error(y_test,base_preds))
        MEAN = y_test.mean()
        """
        Revision: Is the above error metrics, MAE, RMSE, ... shows good or bad ?
        Answer: It depends on all the details, the previous work, what sort of compression strength errors 
                we're trying to obtain, and our domain experience with concrete and slum tests, etc.


        """

        ## Grid Search in Attempt for Better Model
        param_grid = {'C':[0.001,0.01,0.1,0.5,1],
                      'kernel':['linear','rbf','poly'],
                      'gamma':['scale','auto'],
                      'degree':[2,3,4],
                      'epsilon':[0,0.01,0.1,0.5,1,2]}
        # epsilon: is the error we're willing to allow per training data instance

        svr = SVR()
        grid = GridSearchCV(svr,param_grid=param_grid)
        grid.fit(scaled_X_train,y_train)

        optimal_params = grid.best_params_
        grid_preds = grid.predict(scaled_X_test)
        MAE2 = mean_absolute_error(y_test,grid_preds)
        RMSE2 = np.sqrt(mean_squared_error(y_test,grid_preds))


    def fraudulent_wine(self): # Classification Problem
        df = pd.read_csv("./UNZIP_FOR_NOTEBOOKS_FINAL/DATA/wine_fraud.csv")

        quality_unique_values = df['quality'].unique()
        sns.countplot(x='quality', data=df)
        plt.title("1. x=quality")
        plt.show()
        sns.countplot(x='type', hue='quality', data=df)
        plt.title("2. x=type hue=quality")
        plt.show()
        sns.countplot(x='quality', hue='type', data=df)
        plt.title("3. x=quality hue=type")
        plt.show()

        reds = df[df["type"]=='red']
        whites = df[df["type"]=='white']


        fraud_red_percentage = 100* (len(reds[reds['quality']=='Fraud'])/len(reds))
        fraud_white_percentage = 100* (len(whites[whites['quality']=='Fraud'])/len(whites))


        df['quality']= df['quality'].map({'Legit':0,'Fraud':1})
        # df['quality'] = pd.get_dummies(df['quality'],drop_first=True) # same as mapping

        df['type']= df['type'].map({'red':0,'white':1})
        # df['type'] = pd.get_dummies(df['type'],drop_first=True) # same as mapping


        corr_with_quality = df.corr()['quality']

        df.corr()['quality'][:-1].sort_values().plot(kind='bar') # 5. ?
        plt.title("4. df.corr()['quality'].plot[]'bad']")
        plt.show()

        plt.figure(figsize=(3,3))
        sns.clustermap(df.corr(),cmap='viridis', annot=True)
        plt.title("5. clustermap on df.corr()")
        plt.show()

        # Data Analysis is done, let's buil our optimal model via grid searching which takes more time !!!
        X = df.drop('quality',axis=1)
        y = df['quality']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)
        scaler = StandardScaler()
        scaled_X_train = scaler.fit_transform(X_train)
        scaled_X_test = scaler.transform(X_test)
        svc = SVC(class_weight='balanced')
        param_grid = {'C':[0.001,0.01,0.1,0.5,1],'gamma':['scale','auto']}
        grid = GridSearchCV(svc,param_grid)

        grid.fit(scaled_X_train,y_train)

        optimal_params = grid.best_params_
        grid_pred = grid.predict(scaled_X_test)
        confusionMatrix = confusion_matrix(y_test,grid_pred)
        classificationReport = classification_report(y_test,grid_pred)

        display1 = ConfusionMatrixDisplay.from_estimator(grid, scaled_X_test, y_test)
        display1.plot()
        plt.title("6. ConfusionMatrixDisplay RawData")
        plt.show()


        return grid

if __name__ == "__main__":
    svm = SupportVectorMachine()
    svm.classification_SVC()
    svm.regression_SVR()
    svm.fraudulent_wine()







