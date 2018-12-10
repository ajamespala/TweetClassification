pbartotal = 40
import progress as p
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import pandas, numpy, textblob, string
import xgboost
from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
import nltk

# import for NLTK stemming
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

p.progress(0, pbartotal)
# load the dataset
filename = 'data/test.txt'
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

# implement stemming
ps = PorterStemmer()
stem_dict = ["fire", "fires", "firing", "flood", "flooding", "floods", "bombings", "bomb", "bombs", "earthquake", "earthquakes", "exploding", "exploded", "explodes", "explosion", "explosions", "wildfires", "wildfire", "typhoons", "typhoon", "meteorites", "meteors", "meteorite", "meteor", "collapsing", "collapse", "collapses", "haze", "hazing", "hazes", "derails", "derailed", "derailment", "shooting", "shooter", "shootings", "crash", "crashes", "crashing", "crashed", "shot", "shots"] # this isn't right, TOOD: will check

for line in data:
	words = word_tokenize(line)
	for w in words:
		# write to new file instead of printing???
		print(ps.stem(w))

print("Only working for first 2200 entries!! (Change in code)")
labels, texts = [], []
for i, line in enumerate(data.split("\n")):
    content = line.split()
    if(i < 2201):
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
count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
count_vect.fit(trainDF['text'])

p.progress(6, pbartotal)
# transform the training and validation data using count vectorizer object
xtrain_count =  count_vect.transform(train_x)
xvalid_count =  count_vect.transform(valid_x)

p.progress(10, pbartotal)
# load the pre-trained word-embedding vectors 
embeddings_index = {}
for i, line in enumerate(open('data/wiki-news-300d-1M.vec')):
    values = line.split()
    embeddings_index[values[0]] = numpy.asarray(values[1:], dtype='float32')

p.progress(11, pbartotal)
# create a tokenizer 
token = text.Tokenizer()
token.fit_on_texts(trainDF['text'])
word_index = token.word_index

p.progress(12, pbartotal)
# convert text to sequence of tokens and pad them to ensure equal length vectors 
train_seq_x = sequence.pad_sequences(token.texts_to_sequences(train_x), maxlen=70)
valid_seq_x = sequence.pad_sequences(token.texts_to_sequences(valid_x), maxlen=70)

p.progress(13, pbartotal)
# create token-embedding mapping
embedding_matrix = numpy.zeros((len(word_index) + 1, 300))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


p.progress(14, pbartotal)
trainDF['char_count'] = trainDF['text'].apply(len)
trainDF['word_count'] = trainDF['text'].apply(lambda x: len(x.split()))
trainDF['word_density'] = trainDF['char_count'] / (trainDF['word_count']+1)
trainDF['punctuation_count'] = trainDF['text'].apply(lambda x: len("".join(_ for _ in x if _ in string.punctuation))) 
trainDF['title_word_count'] = trainDF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
trainDF['upper_case_word_count'] = trainDF['text'].apply(lambda x: len([wrd for wrd in x.split() if wrd.isupper()]))

p.progress(15, pbartotal)
pos_family = {
    'noun' : ['NN','NNS','NNP','NNPS'],
    'pron' : ['PRP','PRP$','WP','WP$'],
    'verb' : ['VB','VBD','VBG','VBN','VBP','VBZ'],
    'adj' :  ['JJ','JJR','JJS'],
    'adv' : ['RB','RBR','RBS','WRB']
}

p.progress(16, pbartotal)
# function to check and get the part of speech tag count of a words in a given sentence
def check_pos_tag(x, flag):
    cnt = 0
    try:
        wiki = textblob.TextBlob(x)
        for tup in wiki.tags:
            ppo = list(tup)[1]
            if ppo in pos_family[flag]:
                cnt += 1
    except:
        pass
    return cnt

p.progress(17, pbartotal)
trainDF['noun_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'noun'))
trainDF['verb_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'verb'))
trainDF['adj_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'adj'))
trainDF['adv_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'adv'))
trainDF['pron_count'] = trainDF['text'].apply(lambda x: check_pos_tag(x, 'pron'))

p.progress(18, pbartotal)
# train a LDA Model
lda_model = decomposition.LatentDirichletAllocation(n_components=20, learning_method='online', max_iter=20)
X_topics = lda_model.fit_transform(xtrain_count)
topic_word = lda_model.components_ 
vocab = count_vect.get_feature_names()

p.progress(19, pbartotal)
# view the topic models
n_top_words = 10
topic_summaries = []
for i, topic_dist in enumerate(topic_word):
    topic_words = numpy.array(vocab)[numpy.argsort(topic_dist)][:-(n_top_words+1):-1]
    topic_summaries.append(' '.join(topic_words))

p.progress(20, pbartotal)
def train_model(classifier, feature_vector_train, label, feature_vector_valid, is_neural_net=False):
    # fit the training dataset on the classifier
    classifier.fit(feature_vector_train, label)
    
    # predict the labels on validation dataset
    predictions = classifier.predict(feature_vector_valid)
    
    if is_neural_net:
        predictions = predictions.argmax(axis=-1)
    
    return metrics.accuracy_score(predictions, valid_y)
def create_model_architecture(input_size):
    # create input layer 
    input_layer = layers.Input((input_size, ), sparse=True)
    
    # create hidden layer
    hidden_layer = layers.Dense(100, activation="relu")(input_layer)
    
    # create output layer
    output_layer = layers.Dense(1, activation="sigmoid")(hidden_layer)

    classifier = models.Model(inputs = input_layer, outputs = output_layer)
    classifier.compile(optimizer=optimizers.Adam(), loss='binary_crossentropy')
    return classifier 

p.progress(21, pbartotal)
# Naive Bayes on Count Vectors
accuracyNBCV = train_model(naive_bayes.MultinomialNB(), xtrain_count, train_y, xvalid_count)
print ("NB, Count Vectors: " , accuracyNBCV, flush=True)

p.progress(25, pbartotal)
# Linear Classifier on Count Vectors
accuracyLCCV = train_model(linear_model.LogisticRegression(), xtrain_count, train_y, xvalid_count)
print ("LR, Count Vectors: ", accuracyLCCV, flush=True)

p.progress(30, pbartotal)
# RF on Count Vectors
accuracyRFCV = train_model(ensemble.RandomForestClassifier(), xtrain_count, train_y, xvalid_count)
print ("RF, Count Vectors: ", accuracyRFCV, flush=True)


# Extereme Gradient Boosting on Count Vectors
accuracyXGCV = train_model(xgboost.XGBClassifier(), xtrain_count.tocsc(), train_y, xvalid_count.tocsc())
print ("Xgb, Count Vectors: ", accuracyXGCV, flush=True)

p.progress(36, pbartotal)
classifier = create_cnn()
accuracyCNN = train_model(classifier, train_seq_x, train_y, valid_seq_x, is_neural_net=True)
print ("CNN, Word Embeddings: ",  accuracyCNN, flush=True)

p.progress(37, pbartotal)
classifier = create_rnn_lstm()
accuracyRNN = train_model(classifier, train_seq_x, train_y, valid_seq_x, is_neural_net=True)
print ("RNN-LSTM, Word Embeddings: ",  accuracyRNN, flush=True)

p.progress(38, pbartotal)
classifier = create_rnn_gru()
accuracyRNNGRU = train_model(classifier, train_seq_x, train_y, valid_seq_x, is_neural_net=True)
print ("RNN-GRU, Word Embeddings: ",  accuracyRNNGRU, flush=True)

p.progress(39, pbartotal)
classifier = create_bidirectional_rnn()
accuracyRNNBI = train_model(classifier, train_seq_x, train_y, valid_seq_x, is_neural_net=True)
print ("RNN-Bidirectional, Word Embeddings: ",  accuracyRNNBI, flush=True)

p.progress(40, pbartotal)
classifier = create_rcnn()
accuracyRCNN = train_model(classifier, train_seq_x, train_y, valid_seq_x, is_neural_net=True)
print ("RCNN, Word Embeddings: ",  accuracyRCNN, flush=True)


print(flush=True)
print ("NB, Count Vectors: " ,accuracyNBCV)
print ("NB, N-Gram Vectors: ", accuracyNBNG)
print ("NB, CharLevel Vectors: ", accuracyNBCL)
print ("LR, Count Vectors: ", accuracyLCCV)
print ("LR, N-Gram Vectors: ", accuracyLCNG)
print ("LR, CharLevel Vectors: ", accuracyLCCL)
print ("SVM, N-Gram Vectors: ", accuracySVMNG)
print ("RF, Count Vectors: ", accuracyRFCV)
print ("Xgb, Count Vectors: ", accuracyXGCV)
print ("Xgb, CharLevel Vectors: ", accuracyXGCL)
print ("-------------------------------------------")
print ("CNN, Word Embeddings: ",  accuracyCNN)
print ("RNN-LSTM, Word Embeddings: ",  accuracyRNN)
print ("RNN-GRU, Word Embeddings: ",  accuracyRNNGRU)
print ("RNN-Bidirectional, Word Embeddings: ",  accuracyRNNBI)
print ("RCNN, Word Embeddings: ",  accuracyRCNN)
