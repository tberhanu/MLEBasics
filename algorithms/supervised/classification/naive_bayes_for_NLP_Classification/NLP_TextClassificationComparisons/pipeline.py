from sklearn.svm import LinearSVC
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from Visualization_TextVectorization import Visualization_TextVectorization
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer




def report():
    v = Visualization_TextVectorization()
    pipe = Pipeline([('tfidf',TfidfVectorizer(stop_words='english')),('svc',LinearSVC())])
    pipe.fit(v.df['text'],v.df['airline_sentiment'])


    preds = pipe.predict(v.X_test) # Use v.X_test which contains RAW text data B/C pipeline includes vectorization
    print("Pipeline Model Report")

    print(classification_report(v.y_test, preds))


    disp = ConfusionMatrixDisplay.from_estimator(pipe, v.X_test, v.y_test)
    disp.plot()
    plt.show()


    new_tweet = ['good flight']
    print("First prediction: ", pipe.predict(new_tweet))

    new_tweet = ['bad flight']
    print("Second prediction: ", pipe.predict(new_tweet))

    new_tweet = ['ok flight']
    print("Thired prediction: ", pipe.predict(new_tweet))

    
report()