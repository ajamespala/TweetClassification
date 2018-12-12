from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas, numpy, textblob, string
import progress as p
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from keras.models import Sequential
from keras import layers
import keras.constraints

pbartotal = 9

p.progress(0, pbartotal)
# load the dataset
filename = 'data/data15.txt'

# get stop words list
with open('english.txt', 'r') as f:
	stop_words = set(f.read().splitlines())

def remove_stop_words(l, stop_words):
	''' Returns the line without stop words (words that are in the english file) '''
	l = ' '.join([w for w in l.split() if w not in stop_words])
	return l

# remove stop words and call func
with open(filename, 'r') as f:
	content = f.read().splitlines()
	ret = [remove_stop_words(l, stop_words) for l in content]

# write new lines without stop words to a new file
with open('good_tweets.txt', 'w') as f:
	f.write('\n'.join([l for l in ret]))

# data = open('good_tweets.txt').read()  # w/o stop words
data = open(filename).read()  # with stop words
print(filename)
labels, texts = [], []
for i, line in enumerate(data.split("\n")):
    content = line.split()
# text4 -> 4642
# text3 -> 3200
# data24 -> 24158
# data15 and data15unsorted -> 24389
    if(i < 24158):
        labels.append(content[0])
        texts.append(content[1:])
#i = 0
#for label in labels:
#    categories.append([i, label])
#    i++;
#for label in labels:
#    print(label)

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
lb = preprocessing.LabelBinarizer()
train_y = lb.fit_transform(train_y)
print(lb)
print(lb.classes_)
print(train_y)
print('_________________________')
lb = preprocessing.LabelBinarizer()
valid_y = lb.fit_transform(valid_y)
print(lb)
print(lb.classes_)
print(valid_y)

p.progress(5, pbartotal)
# create a count vectorizer object 
count_vect = CountVectorizer(analyzer='word',  stop_words = {'english'}, token_pattern='\w{1,}', max_features = 10000)
count_vect.fit(trainDF['text'])
print(count_vect.get_feature_names())
print(count_vect.get_stop_words())

p.progress(6, pbartotal)
# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)
print(xtrain_count.toarray()[0])

p.progress(9, pbartotal)

input_dim = xtrain_count.shape[1]
print(xtrain_count.shape)
print(input_dim)
model = Sequential()
#model.add(layers.Dense(100, input_dim=input_dim, activation = 'linear', kernel_constraint  = keras.constraints.non_neg()))
model.add(layers.Dense(15, input_dim = input_dim,  activation = 'linear', kernel_constraint  = keras.constraints.non_neg()))

model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit(xtrain_count, train_y, epochs=3, verbose=2, validation_data=(xvalid_count, valid_y), batch_size=32)


print(model.get_weights())
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

#def largest(results):
#    toReturn = []
#    for result in results:
#        biggest = 0;
#        for r in result:
#            if(r > biggest):
#               biggest = r
#        if biggest == 0:
#            return -1
#    

# transform the training and validation data using count vectorizer object
x_to_predict =  count_vect.transform(text1)
result = model.predict(x_to_predict, verbose=1)
print(result[0])

x_to_predict =  count_vect.transform(text2)
result = model.predict(x_to_predict, verbose=1)
print(result[0])

x_to_predict =  count_vect.transform(text3)
result = model.predict(x_to_predict, verbose=1)
print(result[0])
total = 0
for r in result[0]:
    total += r


print(total)
print(lb.classes_)

print(result)

