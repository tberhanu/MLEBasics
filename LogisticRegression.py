import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix,classification_report
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve, RocCurveDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from mpl_toolkits.mplot3d import Axes3D 


class BinaryClassLogisticRegression:
    def __init__(self, visualize=True):
        self.df = pd.read_csv("./UNZIP_FOR_NOTEBOOKS_FINAL/DATA/hearing_test.csv")
        self.visualize = visualize
        self.data_preparation()

    def data_preparation(self):
        X = self.df.drop('test_result',axis=1)
        y = self.df['test_result'] # 0's and 1's, Classification Task

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.1, random_state=101)

        scaler = StandardScaler()
        self.scaled_X_train = scaler.fit_transform(self.X_train)
        self.scaled_X_test = scaler.transform(self.X_test) # we fit only to training data, not test data, to avoid data leakage

    def train_model(self):
        log_model = LogisticRegression()
        log_model.fit(self.scaled_X_train, self.y_train)

        coefficents = log_model.coef_


        # Here, we get Coefficients:  [[-0.95017725  3.46148946]] which means AGE and PHYSICAL_SCORE is +vely and -vely correlated respectively with 'test_result'
        # and we also see that PHYSICAL_SCORE (coeff of 3.46148946) has much greater influence compared to the AGE feature on the 'test_result'
        # Let's try to confirm this via BOX PLOTTING
        if self.visualize:
            plt.figure(dpi=150)
            sns.boxplot(x='test_result', y='physical_score', data=self.df)
            plt.show()

            plt.figure(dpi=150)
            sns.boxplot(x='test_result', y='age', data=self.df)
            plt.show()

        return log_model

    def classification_performance_metrics(self):
        """
        ACCURACY = (TP + TN) / TOTAL
        We shouldn't rely on ACCURACY for Imbalance Classes with only small percentages, instead we should consider other metrics
        like PRECESION, RECALL(=sensitivity), and F1-SCORE..
        RECALL = TP / (total actual +ves) = TP / (TP + FN) >> Recall is focusing on minimizing FALSE NEGATIVES
        PRECISION = TP / (total predicted +ves) = TP / (TP + FP)
        F1-SCORE = (2 * PRECISION * RECALL) / (PRECISION + RECALL)
            Note: If either PRECISION OR RECALL is Zero, the whole F1-SCORE will be Zero.


        """
        log_model = self.train_model()

        y_pred = log_model.predict(self.scaled_X_test) # gives 0's and 1's for 'test_result'
        y_pred_prob = log_model.predict_proba(self.scaled_X_test) # gives the actual probabilities [[prob(0), prob(1)], [prob(0), prob(1)], ....]

        accuracyScore = accuracy_score(self.y_test, y_pred)
        precisionScore = precision_score(self.y_test, y_pred)
        recallScore = recall_score(self.y_test, y_pred)

        confusionMatrix = confusion_matrix(self.y_test, y_pred)
        classificationReport = classification_report(self.y_test, y_pred)


        if self.visualize:
            display1 = ConfusionMatrixDisplay.from_estimator(log_model, self.scaled_X_test, self.y_test) # without Normalizing
            display1.plot()
            plt.title("ConfusionMatrixDisplay RawData")
            plt.show()

            display2 = ConfusionMatrixDisplay.from_estimator(log_model, self.scaled_X_test, self.y_test, normalize='true') # with Normalizing
            display2.plot()
            plt.title("ConfusionMatrixDisplay NormalizedData")
            plt.show()

            """
            Note: 
                While using ConfusionMatrixDisplay:
                    FP and FN have very distinct color from your TP and TN.
                    So, two same color box means they are FALSE POSITIVE and FALSE NEGATIVE.
                    And the other two boxes with different color to each other represents TRUE POSITIVE and TRUE NEGATIVE.

            """


        if self.visualize:
            disp = PrecisionRecallDisplay.from_estimator(log_model, self.scaled_X_test, self.y_test, name="Logistic Regression")
            disp.plot()
            plt.title("PrecisionRecallDisplay")
            plt.show()

            # AUC=Area Under the Curve, ROC=Receiver Operator Characteristic
            disp = RocCurveDisplay.from_estimator(log_model, self.scaled_X_test, self.y_test, name="Logistic Regression: ROC Curve")
            disp.plot()
            plt.title("RocCurveDisplay")
            plt.show()



class MultiClassLogisticRegression:
    def __init__(self):
        self.df = pd.read_csv("/Users/tess/Desktop/MLE2025/projects/UNZIP_FOR_NOTEBOOKS_FINAL/DATA/iris.csv")

    def data_visualization(self):
        unique_species = self.df['species'].unique()
        species_value_counts = self.df['species'].value_counts()

        sns.countplot(self.df['species'])
        plt.title("sns.countplot 'species'")
        plt.show()

        # Checking if any visible correlation between features
        sns.scatterplot(x='sepal_length',y='sepal_width',data=self.df,hue='species')
        plt.title("sns.scatterplot")
        plt.show()

        sns.scatterplot(x='petal_length',y='petal_width',data=self.df,hue='species')
        plt.title("sns.scatterplot")
        plt.show()

        sns.pairplot(self.df,hue='species')
        plt.title("sns.pairplot")
        plt.show()

        #sns.heatmap is to visualize numerical values, so need to drop 'species' which is Object/String
        corr = self.df.drop("species", axis=1).corr()
        sns.heatmap(corr,annot=True)
        plt.title("sns.heatmap")
        plt.show()

        #3D visualization 
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        colors = self.df['species'].map({'setosa':0, 'versicolor':1, 'virginica':2})
        ax.scatter(self.df['sepal_width'], self.df['petal_width'], self.df['petal_length'], c=colors)
        ax.set_xlabel('sepal_width')
        ax.set_ylabel('petal_width')
        ax.set_zlabel('petal_length')
        plt.title("Axes3D")
        plt.show()

    def data_prep_and_train_model(self):
        X = self.df.drop('species',axis=1)
        y = self.df['species']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)
        scaler = StandardScaler()
        scaled_X_train = scaler.fit_transform(X_train)
        scaled_X_test = scaler.transform(X_test)

       
        # Since SIGMOID function gives out 0 or 1, we need to use ovr=One Versus Rest Approach
        # Need to choose the right optimization algorithm (GRADIENT at the background) 
        # via 'solver' like 'liblinear', 'sag', 'saga', .... depending on our datasets size and other needs
        # 'sag' stands for Stochastic Average Gradient'
        # 'max_iter' should be big enough to find for the Gradient Descent to find the Minimum
        #log_model = LogisticRegression(solver='saga',multi_class="ovr",max_iter=5000) # WILL BE DEPRICATED IN VERSION 1.5 FOR MULTI-CLASS
        # base_estimator = LogisticRegression(solver='saga', multi_class="ovr", max_iter=5000) # GIVES WARNING since multi_class='ovr' is redundant for OVR
        base_estimator = LogisticRegression(solver='saga', max_iter=5000)
        log_model = OneVsRestClassifier(estimator=base_estimator)

        # GridSearch for Best Hyper-Parameters
        # Main parameter choices are regularization penalty choice and regularization C value.

        # Penalty Type
        penalty = ['l1', 'l2']
        # l1_ratio = np.linspace(0, 1, 20) # this may raise warning or error saying l1_ration is used only when using elasticnet.
        # Use logarithmically spaced C values (recommended in official docs)
        # 'C' is the lambda term for how strong should this penalty actually be
        # C = np.logspace(0, 4, 10)
        C = np.logspace(0, 10, 20)

        param_grid = {'estimator__penalty': penalty, 'estimator__C': C}
        grid_model = GridSearchCV(log_model, param_grid=param_grid)

        grid_model.fit(scaled_X_train, y_train)

        best_params = grid_model.best_params_

        # print("Optimal Best Parameters: ",  grid_model.best_params_)

        y_pred = grid_model.predict(scaled_X_test)

        accuracyScore = accuracy_score(y_test,y_pred)
        confusionMatrix = confusion_matrix(y_test,y_pred)
        classificationReport = classification_report(y_test,y_pred)

        # print("Accuracy Score: ", accuracy_score(y_test,y_pred))
        # print("Confusion Matrix: \n", confusion_matrix(y_test,y_pred))

        # plot_confusion_matrix(grid_model,scaled_X_test,y_test) #DEPRECATED
        # plot_confusion_matrix(log_model, self.scaled_X_test, self.y_test,normalize='true') #DEPRECATED
        display1 = ConfusionMatrixDisplay.from_estimator(grid_model, scaled_X_test, y_test) # without Normalizing
        display1.plot()
        plt.title("ConfusionMatrixDisplay RawData")
        plt.show()

        display2 = ConfusionMatrixDisplay.from_estimator(grid_model, scaled_X_test, y_test, normalize='true') # with Normalizing
        display2.plot()
        plt.title("ConfusionMatrixDisplay NormalizedData")
        plt.show()


        print("ROC: plot_multiclass_roc, Receiver Operating Characteristic Curve for Multi Classification")
        multiLogReg.plot_multiclass_roc(grid_model, scaled_X_test, y_test, n_classes=3, figsize=(16, 10))


    def plot_multiclass_roc(self, grid_model, X_test, y_test, n_classes, figsize=(5,5)):
        # copied from scikit-learn official page
        y_score = grid_model.decision_function(X_test)

        # structures
        fpr = dict()
        tpr = dict()
        roc_auc = dict()

        # calculate dummies once
        y_test_dummies = pd.get_dummies(y_test, drop_first=False).values
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_dummies[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # roc for each class
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver operating characteristic example')
        for i in range(n_classes):
            ax.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f) for label %i' % (roc_auc[i], i))
        ax.legend(loc="best")
        ax.grid(alpha=.4)
        sns.despine()
        plt.show()

if __name__ == "__main__":
    binaryLogReg = BinaryClassLogisticRegression(False)
    binaryLogReg.classification_performance_metrics()

    multiLogReg = MultiClassLogisticRegression()
    multiLogReg.data_visualization()
    multiLogReg.data_prep_and_train_model()

    












