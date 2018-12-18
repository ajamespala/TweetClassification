# File that reads in specific labels and outputs only those tweets to a new file

def filter_specific_labels(filename, labels, output_file):
	specific_labels = []
	with open (filename, 'r') as f:
		content = f.read().splitlines()
		for i, t in enumerate(content):
			if int(t[9]) in labels or int(t[9:11]) in labels: 
				specific_labels.append('\n' + t)
	with open (output_file, 'w') as f:
		f.write(' '.join([l for l in specific_labels]))


def get_label(l, labels):
	l = ' '.join([w for w in labels if w == t[9]])
	print(l)
	return l


if __name__ == '__main__':
	files = []
	labels = [1, 2, 3,5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
	filename = 'data25.txt'
	output_file = '21labels.txt'
	filter_specific_labels(filename, labels, output_file)
