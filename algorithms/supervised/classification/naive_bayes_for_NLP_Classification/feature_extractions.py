from sklearn.feature_extraction.text import TfidfTransformer,TfidfVectorizer,CountVectorizer
from sklearn.pipeline import Pipeline
from streamlit import text

class FeatureExtraction:

    def count_vectorizer(self, sample_text):
        cv = CountVectorizer(stop_words='english')
        sparse_mat = cv.fit_transform(sample_text)
        print("Count Vectorizer Feature Value:\n", sparse_mat.todense())
        print("Shape of Sparse Matrix:", sparse_mat.shape)
        print("Feature Names:\n", cv.get_feature_names_out())
        print("Vocabulary:\n", cv.vocabulary_)

    def tfidf_transformer(self, sample_text):
        tfidf_transformer = TfidfTransformer()
        cv = CountVectorizer(stop_words='english')
        counts = cv.fit_transform(sample_text)
        X_tfidf = tfidf_transformer.fit_transform(counts)
        print("TF-IDF Output:\n", X_tfidf.toarray())
        print("Shape of TF-IDF Matrix:", X_tfidf.shape)

    def pipeline(self, sample_text):
        pipe = Pipeline([('cv',CountVectorizer(stop_words='english')),('tfidf',TfidfTransformer())])
        results = pipe.fit_transform(sample_text)
        print("Pipeline TF-IDF Output:\n", results.toarray())

    def tfidf_vectorizer(self, sample_text):
        # Does both above in a single step!
        tfidf_vectorizer = TfidfVectorizer(stop_words='english')
        X_tfidf = tfidf_vectorizer.fit_transform(sample_text)
        print("TF-IDF Vectorizer Output:\n", X_tfidf.toarray())
        print("Shape of TF-IDF Vectorizer Matrix:", X_tfidf.shape)
        print("Feature Names:\n", tfidf_vectorizer.get_feature_names_out())

if __name__ == "__main__":
    sample_text = [
        "I love programming in Python",
        "Python is a great programming language",
        "I enjoy learning new programming languages"
    ]

    feature_extractor = FeatureExtraction()
    feature_extractor.count_vectorizer(sample_text)
    print("================================")
    feature_extractor.tfidf_transformer(sample_text)
    print("================================")
    feature_extractor.pipeline(sample_text)
    print("================================")
    feature_extractor.tfidf_transformer(sample_text)