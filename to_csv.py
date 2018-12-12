from os import walk

def main(files):
	with open ('english.txt', 'r') as f:
		bad_words = set(f.read().splitlines())
	filtered_by_file = filter_bad_words(bad_words, files)	
	labeled_data = label(filtered_by_file)

def label(tweets):
	''' return single list of tweets ''' 
	for t in tweets:
		if t[:9] != '__label__':
			print(t)
			print('hello')
			continue
		print(t[11:] + ',' + t[9])

def remove_bad_words(l, bad_words):
	# l = ' '.join([w for w in l.split() if w not in bad_words and w.isalnum()]) #  and its a word
	l = ' '.join([w for w in l.split() if w not in bad_words]) #  and its a word
	return l


def filter_bad_words(bad_words, files):
	good_tweets = []
	for filename in files:
		with open(data_folder + '/' + filename, 'r') as f:
			content = f.read().splitlines() # list of strings
			print(content[0])
			good_tweets.append([remove_bad_words(l, bad_words) for l in content])
	return [item for sublist in good_tweets for item in sublist]


if __name__ == '__main__':
	files = []
	data_folder = 'old_text'
	for (dirpath, dirnames, filenames) in walk(data_folder):
		files.append(filenames)
	files = [f for sub in files for f in sub]
	main(files)
