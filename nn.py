from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas, textblob, string
import numpy as np
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from keras.models import Sequential
from keras import layers
import keras.constraints


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


def classify(text):
	''' classifies individual tweets '''
	text = str(text)
	text = [text]
	x_to_predict =  count_vect.transform(text)
	result = model.predict(x_to_predict, verbose=0)
	# find max result[0]
	label_index = np.argmax(result[0])	
	return label_index


if __name__ == '__main__':
	# load the dataset
	original_data = raw_input('Enter the filename for the data set: ')
	#change filename for train/test set
	output_file = 'good_tweets.txt'

	# get stop words list
	with open('english.txt', 'r') as f:
		stop_words = set(f.read().splitlines())

	create_new_file(original_data, stop_words, output_file)

	# setting filename to the file without stop words
	filename = output_file
	data = open(filename).read()  
	labels, texts = [], []

	for i, line in enumerate(data.split("\n")):
		content = line.split()
		if content:
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
	lb = preprocessing.LabelBinarizer()
	valid_y = lb.fit_transform(valid_y)

	# create a count vectorizer object 
	count_vect = CountVectorizer(analyzer='word',  stop_words = {'english'}, token_pattern='\w{1,}', max_features = 10000)
	count_vect.fit(trainDF['text'])

	# transform the training and validation data using count vectorizer object
	xtrain_count =  count_vect.transform(train_x)
	xvalid_count =  count_vect.transform(valid_x)

	#create model, and compile
	input_dim = xtrain_count.shape[1]
	model = Sequential()
	numCategories = input('Enter the number of categories: ')
	model.add(layers.Dense(100, input_dim = input_dim,  activation = 'linear', kernel_constraint  = keras.constraints.non_neg()))
	model.add(layers.Dense(numCategories, input_dim = input_dim,  activation = 'linear', kernel_constraint  = keras.constraints.non_neg()))

	model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

	history = model.fit(xtrain_count, train_y, epochs=3, verbose=2, validation_data=(xvalid_count, valid_y), batch_size=32)

	loss, accuracy = model.evaluate(xtrain_count, train_y, verbose=True)
	print("Training Accuracy: {:.6f}".format(accuracy))
	loss, accuracy = model.evaluate(xvalid_count, valid_y, verbose=True)
	print("Testing Accuracy:  {:.6f}".format(accuracy))

	test_option = raw_input('Would you like to validate with new data? (yes/no) ')
	if test_option == "yes":
		validation_file = raw_input('Enter the filename for the validation set: ')
		output_file2 = 'good_tweets_2.txt'
		# create a copy of validation file w/o stop words
		create_new_file(validation_file, stop_words, output_file2)
		filename = output_file2
		data = open(filename).read()  

		labels, texts = [], []
		print("Validating...")
		num_correct = 0
		for i, line in enumerate(data.split("\n")):
			content = line.split()
			if content:
				label_index = classify(content[1:])
				#print("It is classified as " + lb.classes_[label_index] + ", should be " + content[0])
				if (str(content[0]) == str(lb.classes_[int(label_index)])):
					num_correct = num_correct + 1
		#numEntries = input('Enter in the number of entries: ')
		numEntries = sum(1 for line in open(filename))

		print("Test: " + str(num_correct) + " out of " + str(numEntries) + " correct")
	else:
		exit(0)
