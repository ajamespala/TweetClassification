files needed:
 https://gist.github.com/kunalj101/ad1d9c58d338e20d09ff26bcc06c4235
 https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki-news-300d-1M.vec.zip

(both are stored in data/)

dependencies:
 from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
 from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
 from sklearn import decomposition, ensemble
 import pandas, numpy, textblob, string
 import xgboost
 from keras.preprocessing import text, sequence
 from keras import layers, models, optimizers
 import nltk
 nltk.download('punkt')
 nltk.download('averaged_perceptron_tagger')
