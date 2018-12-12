from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas, numpy, textblob, string
import progress as p
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from keras.models import Sequential
from keras import layers
import keras.constraints


# load the dataset
filename = 'data/test.txt'
# data = open(filename).read()
print(filename)

# get stop words list
with open ('english.txt', 'r') as f:
	stop_words = set(f.read().splitlines())

def remove_stop_words(l, stop_words):
	''' Returns the line without   '''
	l = ' '.join([w for w in l.split() if w not in bad_words])
	return l


# remove stop words
with open(filename, 'r') as f:
	content = f.read().splitlines()
	ret = [remove_stop_words(l, stop_words) for l in content]
with open('good_tweets.txt', 'w') as f:
	f.write('\n'.join([l for l in ret]))
data = open('good_tweets.txt').read()

print("Only working for first 2200 entries!! (Change in code)")
labels, texts = [], []
for i, line in enumerate(data.split("\n")):
    content = line.split()
# text4 -> 4642
# text3 -> 3200
    if(i < 4612):
        labels.append(content[0])
        texts.append(content[1:])

# create a dataframe using texts and lables
trainDF = pandas.DataFrame()
texts1=[' '.join(line) for line in texts]
trainDF['text'] = texts1
trainDF['label'] = labels

#split the data into training and testing data
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])


# label encode the target variable 
lb = preprocessing.LabelBinarizer()
train_y = lb.fit_transform(train_y)
print(lb.classes_)
lb = preprocessing.LabelBinarizer()
valid_y = lb.fit_transform(valid_y)

# create a count vectorizer object for inputs 
count_vect = CountVectorizer(analyzer='word',  stop_words = {'english'}, token_pattern=r'\w{1,}', max_features = 2000)
count_vect.fit(trainDF['text'])

# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)


input_dim = xtrain_count.shape[1]
model = Sequential()
#model.add(layers.Dense(20, input_dim=input_dim, activation = 'linear', kernel_constraint  = keras.constraints.non_neg()))
model.add(layers.Dense(4, input_dim = input_dim,  activation = 'linear', kernel_constraint  = keras.constraints.non_neg()))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(xtrain_count, train_y, epochs=1, verbose=1, validation_data=(xvalid_count, valid_y), batch_size=1000)


weights = model.get_weights()
loss, accuracy = model.evaluate(xtrain_count, train_y, verbose=False)
print("Training Accuracy: {:.6f}".format(accuracy))
loss, accuracy = model.evaluate(xvalid_count, valid_y, verbose=False)
print("Testing Accuracy:  {:.6f}".format(accuracy))

#samples = []
#texts = []
#samples.append("Meteor injures 725 in Russia: The meteor streaked through the skies over Russia's southern Chelyabinsk region")
#for sample in samples:
#content =sample.split()
#texts.append(content[1:])
    
#trainDF = pandas.DataFrame()
#texts1=[' '.join(line) for line in texts]

# transform the training and validation data using count vectorizer object
#x_to_predict =  count_vect.transform(text1)
#result = model.predict(x_to_predict, verbose=1)
#print(result[0])
#
#x_to_predict =  count_vect.transform(text2)
#result = model.predict(x_to_predict, verbose=1)
#print(result[0])
#
#x_to_predict =  count_vect.transform(text3)
#result = model.predict(x_to_predict, verbose=1)
#print(result[0])
