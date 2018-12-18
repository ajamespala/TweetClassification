from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas, textblob, string
import numpy as np
import progress as p
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from keras.models import Sequential
from keras import layers
import keras.constraints

pbartotal = 9

# load the dataset
filename = 'data/ThisIsTheOne.txt'
testfile = 'data/4missing.txt'
output_file = 'good_tweets.txt'
output_file2 = 'good_tweets_2.txt'

# get stop words list
with open('english.txt', 'r') as f:
	stop_words = set(f.read().splitlines())

def remove_stop_words(l, stop_words):
	''' Returns the line without stop words (words that are in the english file) '''
	l = ' '.join([w for w in l.split() if w not in stop_words])
	return l


def create_new_file(filename, stop_words, output_file):
	# remove stop words and call func
	with open(filename, 'r') as f:
		content = f.read().splitlines()
		ret = [remove_stop_words(l, stop_words) for l in content]

	# write new lines without stop words to a new file
	with open(output_file, 'w') as f:
		f.write('\n'.join([l for l in ret]))

create_new_file(filename, stop_words, output_file)
create_new_file(testfile, stop_words, output_file2)

# setting filename to the file without stop words
filename = 'good_tweets.txt'
data = open(filename).read()  
labels, texts = [], []
# text4 -> 4642
# text3 -> 3200
# data24 -> 24158
# data15 and data15unsorted -> 24389
# sentiment -> 988
numEntries = 22250
for i, line in enumerate(data.split("\n")):
    content = line.split()
    if(i < int(numEntries)):
        labels.append(content[0])
        texts.append(content[1:])
# create a dataframe using texts and lables
trainDF = pandas.DataFrame()
texts1=[' '.join(line) for line in texts]
trainDF['text'] = texts1
trainDF['label'] = labels
train_x, valid_x, train_y, valid_y = model_selection.train_test_split(trainDF['text'], trainDF['label'])
# label encode the target variable 
lb = preprocessing.LabelBinarizer()
train_y = lb.fit_transform(train_y)
#print(lb)
print(lb.classes_)
#print(train_y)
print('_________________________')
lb = preprocessing.LabelBinarizer()
valid_y = lb.fit_transform(valid_y)
# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word',  stop_words = {'english'}, token_pattern='\w{1,}', max_features = 10000)
count_vect.fit(trainDF['text'])
#print(count_vect.get_feature_names())
#print(count_vect.get_stop_words())
print(train_x)
# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)

input_dim = xtrain_count.shape[1]
#print(xtrain_count.shape)
#print(input_dim)
model = Sequential()
numCategories = 14
#model.add(layers.Dense(100, input_dim = input_dim,  activation = 'linear', kernel_constraint  = keras.constraints.non_neg()))
model.add(layers.Dense(numCategories, input_dim = input_dim,  activation = 'linear', kernel_constraint  = keras.constraints.non_neg()))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(xtrain_count, train_y, epochs=3, verbose=2, validation_data=(xvalid_count, valid_y), batch_size=32)

#print(model.get_weights())
weights = model.get_weights()
for layer in model.layers:
    print(layer.name,layer.input_shape, layer.output_shape)
loss, accuracy = model.evaluate(xtrain_count, train_y, verbose=False)
print("Training Accuracy: {:.6f}".format(accuracy))
loss, accuracy = model.evaluate(xvalid_count, valid_y, verbose=False)
print("Testing Accuracy:  {:.6f}".format(accuracy))

#samples = []
#texts = []
#samples.append("Meteor injures 725 in Russia: The meteor streaked through the skies over Russia's southern Chelyabinsk region")
#for sample in samples:
text1 = ['Meteor injures 725 in Russia: The meteor streaked through the skies over Russia\'s southern Chelyabinsk region']
text2 = ['New York train crash probe begins: The US authorities begin an investigation into the causes of Sunday\'s New Y... http://t.co/RS6ku9393B']
text3 = ['The train crash was horrible. Praying for the victims']
#content =sample.split()
#texts.append(content[1:])
    
#trainDF = pandas.DataFrame()
#texts1=[' '.join(line) for line in texts]

def func(text):
	text = str(text)
	text = [text]
	print(type(text))
	x_to_predict =  count_vect.transform(text)
	result = model.predict(x_to_predict, verbose=1)
	# fina max result[0]
	label_index = np.argmax(result[0])	
	return label_index


def test(testDF, lb):
	#labels = # convert pandas data frame to
	text = testDF['text'] 
	labels = testDF['label']
	labels = lb.transform(labels)
	num_correct = 0
	for i, t in enumerate(text):
		ind = func(t)
		print(t)
		print(str(ind))
		if ind == i:
			num_correct = num_correct + 1
	return num_correct

# transform the training and validation data using count vectorizer object
print(func(text1))
#x_to_predict =  count_vect.transform(text1)
#result = model.predict(x_to_predict, verbose=1)

print(func(text2))
print(func(text3))

#print(lb.classes_)


# TODO: call on data set to get accuracy score	

# setting filename to the file without stop words
filename = output_file2
data = open(filename).read()  
labels, texts = [], []
# text4 -> 4642
# text3 -> 3200
# data24 -> 24158
# data15 and data15unsorted -> 24389
# sentiment -> 988
numEntries = 3684
for i, line in enumerate(data.split("\n")):
    content = line.split()
    if(i < int(numEntries)):
        labels.append(content[0])
        texts.append(content[1:])
# create a dataframe using texts and lables
testDF = pandas.DataFrame()
texts1=[' '.join(line) for line in texts]
testDF['text'] = texts1
testDF['label'] = labels

numCorrect = test(testDF, lb)

print("Test: " + str(numCorrect) + " out of " + str(numEntries)+ "correct")
