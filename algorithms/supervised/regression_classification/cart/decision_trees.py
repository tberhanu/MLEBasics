import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,classification_report,ConfusionMatrixDisplay
from sklearn.tree import plot_tree


class DecisionTrees:
    def __init__(self):
        """
        Note: If your categorical features are strings or objects, must convert them to numbers, where ORDINAL ENCODING (assigning integers to categories)
              is better than ONE-HOT ENCODING in order to avoid unnecessary complexity. (from sklearn.preprocessing import OrdinalEncoder)
              
        *** Tree based models like DecisionTreeClassifier, RandomForest, and GradientBoosting are OK if strings converted to numbers via
            OrdinalEncoder or OneHotEncoder, but the modeliteslef doesn't need the features to be scaled or normalized. 
            
        Here, we skip grid searching for optimal params as the evaluation metrics shows good results,
        but for larger datasets in real life where classifying is more of a challenge, we may need gridsearching.

        Note: Decision Trees are very prone to overfitting, especially when they are allowed to grow deep and complex without constraints.
        
        How to Prevent Overfitting
            Limit Tree Depth: Use max_depth to restrict how deep the tree can grow.

            Minimum Samples per Leaf: Set min_samples_leaf to ensure each leaf has enough data.

            Pruning: Remove branches that have little predictive power.

            Use Ensemble Methods:

                Random Forests: Average multiple trees to reduce variance.
    
                Gradient Boosting: Builds trees sequentially to correct errors.

        Note: For Tree Based methods like a Single Decision Tree or a Random Forest, no need to worry about having to scale or
              standardize or normalize any of the feature data points since we're just going to be doing a split based on a
              single feature.



        Ideal Dataset Characteristics for Decision Trees
            1. Categorical Features: Decision Trees handle categorical variables naturally without needing one-hot encoding. 

            2. Non-linear Relationships: They’re great when the relationship between features and target isn’t linear. 

            3. Interpretable Rules Needed: If you need a model that’s easy to explain, Decision Trees provide clear decision paths. 

            4. Missing Values or No Need for Scaling: Trees can handle missing data and don’t require feature normalization. 

            5. Small to Medium-Sized Datasets: Trees can overfit large datasets unless pruned or regularized. 

        Why Decision Trees Are Non-Parametric
            1. No assumptions: Trees don’t assume the data follows a linear trend, a normal distribution, or any specific shape.

            2. Structure adapts to data: The number of splits, depth of the tree, and branching structure are all determined by the data itself.

            3. Can model complex relationships: Trees can capture non-linear patterns, interactions between features, and handle mixed data types (numerical + categorical).

        """
        pass

    def basic_decision_tree(self):
        

        df = pd.read_csv("../../../../files/final_files/DATA/penguins_size.csv")
        species_types = df['species'].unique()
        size_befor_drop = len(df)

        ### Missing Data
        na_df = df.isna()
        na_df2 = df.isnull()
        na_sum_per_column_df = df.isna().sum()
        na_sum_all_df = df.isna().sum().sum()


        # What percentage are we dropping?
        na_sex_percentage = 100*(10/344)

        df = df.dropna() # dropping any rows that have missing values
        size_after_drop = len(df)

        df['sex'].unique()
        df['island'].unique()

        missed_sex_rows = df[df['sex']=='.'] # let's fix this unknown sex type

        stat1 = df[df['species'] == 'Gentoo'].groupby('sex').describe()
        stat2 = df[df['species'] == 'Gentoo'].groupby('sex').describe().transpose()
        # based on this statistics, try to decide which sex type is closer to that row=336
        
        df.at[336, 'sex'] = 'FEMALE'
        fixed_row = df.loc[336]



        ## Visualization
        sns.scatterplot(x='culmen_length_mm',y='culmen_depth_mm',data=df,hue='species',palette='Dark2')
        plt.title("scatterplot")
        plt.show()
        sns.pairplot(df,hue='species',palette='Dark2') # shows clear separation so any ML algo should be good to do the classification
        plt.title("paiplot")
        plt.show()
        sns.catplot(x='species',y='culmen_length_mm',data=df,kind='box',palette='Dark2')
        plt.title("catplot kind=box")
        plt.show()
        sns.catplot(x='species',y='culmen_length_mm',data=df,kind='box',col='sex',palette='Dark2')
        plt.title("catplot kind=box col=sex")
        plt.show()


        ## Feature Engineering

        # pd.get_dummies(df) # this will convert ALL including the label, 'species'
        # pd.get_dummies(df.drop('species',axis=1),drop_first=True) # dropping 'species' first before get dummies

        ## Train | Test Split
        X = pd.get_dummies(df.drop('species',axis=1),drop_first=True) # dropping 'species' first before get dummies
        y = df['species']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
        # Note: We don't need SCALING in Decision Trees as we're just compaing a single feature along a particular range of values !!!

        model = DecisionTreeClassifier() # with default hyperparameter values
        model.fit(X_train,y_train)
        base_preds = model.predict(X_test)

        ## Evaluation
        confusionMatrix = confusion_matrix(y_test,base_preds)
        # plot_confusion_matrix(model,X_test,y_test)
        display1 = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test) # without Normalizing
        display1.plot()
        plt.title("ConfusionMatrixDisplay RawData")
        plt.show()

        classificationReport = classification_report(y_test,base_preds)

        # Good thing about Decision Tree is it's very interpretable by humans
        feature_importances = model.feature_importances_ # relative importance of features in the decision making
        # feature importances order align with the column/feature of our df, but we can still make another df to clearly display
        feature_importances_df = pd.DataFrame(index=X.columns,data=feature_importances,columns=['Feature Importance'])
        feature_importances_df_sorted = feature_importances_df.sort_values('Feature Importance', ascending=False)
        # You may see that some of the feature's has Zero importance, it's b/c the algo may not consider them to split the tree
        # unless you really force it to do so which may result in overfitting.


        sns.boxplot(x='species',y='body_mass_g',data=df)
        plt.title("boxplot")

        ## Visualize the Tree
        plt.figure(figsize=(12,8))
        plot_tree(model)
        plt.title("plot_tree")
        plt.show()

        plt.figure(figsize=(12,8),dpi=150)
        plot_tree(model,filled=True,feature_names=X.columns)
        plt.title("with filled=True & feature_names")
        plt.show()

        # help(DecisionTreeClassifier)

        pruned_tree = DecisionTreeClassifier(max_depth=2)
        pruned_tree.fit(X_train,y_train)
        title = "pruned max_depth=2"
        self.report_model(pruned_tree, X_test, y_test, title, X.columns)

        ## Max Leaf Nodes
        pruned_tree = DecisionTreeClassifier(max_leaf_nodes=3)
        pruned_tree.fit(X_train,y_train)
        title = "pruned max_leaf_nodes=3"
        self.report_model(pruned_tree, X_test, y_test, title, X.columns)

        ## Criterion
        entropy_tree = DecisionTreeClassifier(criterion='entropy')
        entropy_tree.fit(X_train,y_train)
        title = "criterion=entropy"
        self.report_model(entropy_tree, X_test, y_test, title, X.columns)

        

    def report_model(self, model, X_test, y_test, title, X_columns):
        ## Reporting Model Results
        #To begin experimenting with hyperparameters, let's create a function that reports back classification results and plots out the tree.

        model_preds = model.predict(X_test)
        print(classification_report(y_test,model_preds))
        print('\n')
        plt.figure(figsize=(12,8),dpi=150)
        plot_tree(model,filled=True,feature_names=X_columns)
        plt.title(title)
        plt.show()





if __name__ == "__main__":
    dt = DecisionTrees()
    dt.basic_decision_tree()
