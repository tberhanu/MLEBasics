"""
NLP and Supervised Learning
Classification of Text Data
The Data
Source: https://www.kaggle.com/crowdflower/twitter-airline-sentiment?select=Tweets.csv

This data originally came from Crowdflower's Data for Everyone library.

As the original source says,

A sentiment analysis job about the problems of each major U.S. airline. Twitter data was scraped from 
February of 2015 and contributors were asked to first classify positive, negative, and neutral tweets, 
followed by categorizing negative reasons (such as "late flight" or "rude service").

The Goal: Create a Machine Learning Algorithm that can predict if a tweet is positive, neutral, 
or negative. In the future we could use such an algorithm to automatically read and flag tweets 
for an airline for a customer service agent to reach out to contact.¶
 """




from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..")) 
sys.path.append(parent_dir)

from Visualization_TextVectorization import Visualization_TextVectorization


class LinearSVCModel:
    def __init__(self):
        pass

    def report():
        tfidf = Visualization_TextVectorization()

        svc = LinearSVC() # Linear Kernel SVC (faster than RBF SVC)
        svc.fit(tfidf.X_train_tfidf,tfidf.y_train)


        preds = svc.predict(tfidf.X_test_tfidf)
        print("SVC Model Report")

        print(classification_report(tfidf.y_test, preds))


        disp = ConfusionMatrixDisplay.from_estimator(svc, tfidf.X_test_tfidf, tfidf.y_test)
        disp.plot()
        plt.show()


if __name__ == "__main__":
    LinearSVCModel.report()
