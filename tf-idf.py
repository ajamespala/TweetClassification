from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas, numpy, textblob, string
import progress as p
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm

pbartotal = 9

p.progress(0, pbartotal)
# load the dataset
filename = 'data/test.txt'
data = open(filename).read()
print(filename)
print("Only working for first 3202 entries!! (Change in code)")
labels, texts = [], []
for i, line in enumerate(data.split("\n")):
    content = line.split()
    if(i < 3202):
        labels.append(content[0])
        texts.append(content[1:])

p.progress(2, pbartotal)
# create a dataframe using texts and lables
trainDF = pandas.DataFrame()
texts1=[' '.join(line) for line in texts]
trainDF['text'] = texts1
trainDF['label'] = labels

p.progress(3, pbartotal)
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])

p.progress(4, pbartotal)
# label encode the target variable 
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)


p.progress(5, pbartotal)
# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word',  stop_words = {'english'}, token_pattern=r'\w{1,}', max_features = 100)
count_vect.fit(trainDF['text'])
print(count_vect.get_feature_names())
print(count_vect.get_stop_words())

p.progress(6, pbartotal)
# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)
print(xtrain_count.toarray()[0])


p.progress(7, pbartotal)
# word level tf-idf
tfidf_vect = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', stop_words = {'english'}, max_features=500)
tfidf_vect.fit(trainDF['text'])
#xtrain_tfidf =  tfidf_vect.transform(train_x)
#xvalid_tfidf =  tfidf_vect.transform(valid_x)

p.progress(8, pbartotal)
# ngram level tf-idf 
tfidf_vect_ngram = TfidfVectorizer(analyzer='word', token_pattern=r'\w{1,}', ngram_range=(2,3), stop_words = {'english'}, max_features=500)
tfidf_vect_ngram.fit(trainDF['text'])
xtrain_tfidf_ngram =  tfidf_vect_ngram.transform(train_x)
xvalid_tfidf_ngram =  tfidf_vect_ngram.transform(valid_x)
#print(tfidf_vect.get_feature_names())
#print(xtrain_tfidf)
#print (xtrain_tfidf_ngram.shape)
#print (xvalid_tfidf_ngram.shape)
#print(tfidf_vect.get_stop_words())

p.progress(9, pbartotal)
# characters level tf-idf
tfidf_vect_ngram_chars = TfidfVectorizer(analyzer='char', token_pattern=r'\w{1,}', stop_words = {'english'}, ngram_range=(2,3), max_features=500)
tfidf_vect_ngram_chars.fit(trainDF['text'])
xtrain_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(train_x)
xvalid_tfidf_ngram_chars =  tfidf_vect_ngram_chars.transform(valid_x)
#print(tfidf_vect_ngram_chars.get_stop_words())
