# File that reads in specific labels and outputs only those tweets to a new file

def filter_specific_labels(filename, labels, output_file):
	specific_labels = []
	with open (filename, 'r') as f:
		content = f.read().splitlines()
		for i, t in enumerate(content):
			if int(t[9:11]) in labels: 
				specific_labels.append('\n' + t)
	with open (output_file, 'a') as f:
		f.write(' '.join([l for l in specific_labels]))


def get_label(l, labels):
	l = ' '.join([w for w in labels if w == t[9]])
	print(l)
	return l


if __name__ == '__main__':
	files = []
	labels = [2]
	filename = 'original.txt'
	output_file = 'validate.txt'
	filter_specific_labels(filename, labels, output_file)
