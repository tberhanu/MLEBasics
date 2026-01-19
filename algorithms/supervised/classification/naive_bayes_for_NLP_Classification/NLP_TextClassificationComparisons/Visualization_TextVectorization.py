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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

class Visualization_TextVectorization:
    def __init__(self):
        self.df = pd.read_csv("/Users/tess/Desktop/MLE2025/ML-Masterclass/UNZIP_FOR_NOTEBOOKS_FINAL (1)/DATA/airline_tweets.csv")
        self.X_train_tfidf, self.X_test_tfidf, self.y_train, self.y_test = self.train_test_split_and_tfidf()

    def plotting(self):
        sns.countplot(data=self.df,x='airline',hue='airline_sentiment')
        plt.show()

        sns.countplot(data=self.df,x='negativereason')
        plt.xticks(rotation=90);
        plt.show()

        sns.countplot(data=self.df,x='airline_sentiment')
        plt.show()

    def train_test_split_and_tfidf(self):
        # df['airline_sentiment'].value_counts()

        data = self.df[['airline_sentiment','text']]

        y = data['airline_sentiment']
        X = data['text']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)
        self.X_test = X_test  # Store X_test for pipeline use
        tfidf = TfidfVectorizer(stop_words='english')

        tfidf.fit(X_train)

        X_train_tfidf = tfidf.transform(X_train)
        X_test_tfidf = tfidf.transform(X_test)
        
        return X_train_tfidf, X_test_tfidf, y_train, y_test