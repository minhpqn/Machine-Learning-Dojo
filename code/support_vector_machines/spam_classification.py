""" 
	Machine Learning Online Class
    Exercise 6 | Spam Classification with SVMs
"""

import os
import sys
import numpy as np
from scipy.io import loadmat
from nltk import stem
from sklearn import svm
import re

def getVocabulary():
	""" Load vocabulary in the file vocab.txt into a dictionary
	Return
	---------
	vocab : dict
	"""

	vocab = {}
	f = open('vocab.txt', 'rU')
	for line in f:
		line = line.strip()
		if line == '': continue
		_id, w = line.split('\t')
		vocab[w] = int(_id)
	f.close()

	return vocab

def process_email(filename):
	""" Preprocess email content

	 Lower-casing
	 Stripping HTML
	 Normalizing URLS
	 Normalizing Email Addresses
	 Normalizing Numbers
	 Normalizing Dollars
	 Word Stemming
	 Removal of non-words

	Return
	-----------
	words : list
	    Words in the email
	word_indices : list
	    Indices of words in email if words is in the list vocab.txt
	"""

	vocab = getVocabulary()

	stemmer = stem.PorterStemmer()
	word_indices = []
	words = []
	f = open(filename, 'rU')
	for line in f:
		line = line.strip()
		# Lower-casing
		line = line.lower()
		# Stripping HTML
		line = re.sub(r'<[^<>]+>', ' ', line)
		# Handle Numbers
		# Look for one or more characters between 0-9
		line = re.sub(r'[0-9]+', 'number', line)
		# Normalizing URLS
		line = re.sub(r'(http|https)://[^\s]*', 'httpaddr', line)
		# Normalizing Email Addresses
		line = re.sub(r'[^\s]+@[^\s]+', 'emailaddr', line)
		# Handle $ sign
		line = re.sub(r'[$]+', 'dollar', line)

		for w in re.split(r'\W+', line):
			_w = re.sub(r'[^a-zA-Z0-9]', '', w)
			# https://docs.python.org/2/tutorial/errors.html
			try: 
				_w = stemmer.stem(_w)
			except Exception:
				_w = ''

			if len(_w) < 1:
				continue

			words.append(_w)

			if vocab.has_key(_w):
				word_indices.append( vocab[_w] )
	f.close()
	return word_indices, words

def emailFeatures(word_indices):
	""" Return feature vector from word indices of an email
	Parameters:
	------------
	word_indices : list
	    List of integer word indices

	Return
	------------
	features : numpy.ndarray
	    Feature vector
	"""

	vocab = getVocabulary()
	features = np.zeros( len(vocab) )
	for i in word_indices:
		features[i-1] = 1.0

	return features.reshape(1,-1)

def printEmail(words):
	""" Print content of an email. Print newline character if a line
	exceeds 78 characters
	"""
	l = 0
	for w in words:
		if l + len(w) + 1 > 78:
			print
			l = len(w)
		else:
			l += len(w) + 1

		sys.stdout.write('%s ' % w)
	print

if __name__ == '__main__':
	os.system('cls' if os.name == 'nt' else 'clear')
	# ========== Part 1: Email Preprocessing ==================
	# To use an SVM to classify emails into Spam v.s. Non-Spam, 
	# you first need to convert each email into a vector of features. 
	# In this part, you will implement the preprocessing steps 
	# for each email. You should complete the code in processEmail.m 
	# to produce a word indices vector for a given email.

print('\nPreprocessing sample email (emailSample1.txt)');
word_indices, words = process_email('emailSample1.txt')
print '\n==== Processed Email ====\n'
printEmail(words)
print '\n\n========================='
print 'Word Indices:'
print ' '.join( [ str(i) for i in word_indices] )
raw_input('<Press Enter to continue>\n')

# ==================== Part 2: Feature Extraction ====================
#   Now, you will convert each email into a vector of features in R^n. 
#   You should complete the code in emailFeatures.m to produce a feature
#   vector for a given email.

print '\nExtracting features from sample email (emailSample1.txt)'

word_indices, words = process_email('emailSample1.txt')
features = emailFeatures(word_indices)
print('Length of feature vector: %d' % features.size)
print('Number of non-zero entries: %d' % np.sum(features > 0))
raw_input('<Press Enter to continue>\n')

# =========== Part 3: Train Linear SVM for Spam Classification ========
#  In this section, you will train a linear classifier to 
#  determine if an email is Spam or Not-Spam.

data = loadmat('spamTrain.mat')
X = data['X']
y = data['y']
y = y.flatten()

print 'Shape of X and y'
print X.shape
print y.shape

print '\nTraining Linear SVM (Spam Classification)'
C = 0.1
clf = svm.SVC(C, kernel='linear')
clf.fit(X, y)
print('Training Accuracy: %f' % clf.score(X, y))

# =================== Part 4: Test Spam Classification ================
#  After training the classifier, we can evaluate it on a test set. We have included a test set in spamTest.mat
data  = loadmat('spamTest.mat')
Xtest = data['Xtest']
ytest = data['ytest']
ytest = ytest.flatten()

print 'Shape of Xtest and ytest'
print Xtest.shape
print ytest.shape
print('Testing Accuracy: %f' % clf.score(Xtest, ytest))

# ========= (Optional): Try to use linearSVC ==============
print '\nUsing svm.LnearSVC for classification'
linear_clf = svm.LinearSVC(C=C)
linear_clf.fit(X, y)
print('Training Accuracy: %f' % linear_clf.score(X, y))
print('Testing Accuracy: %f' % linear_clf.score(Xtest, ytest))
raw_input('<Press Enter to continue>\n')

# ================= Part 5: Top Predictors of Spam ====================
#  Since the model we are training is a linear SVM, we can inspect the
# weights learned by the model to understand better how it is determining
# whether an email is spam or not. The following code finds the words with
# the highest weights in the classifier. Informally, the classifier
# 'thinks' that these words are the most likely indicators of spam.

vocab = getVocabulary()
new_vocab = { idx:w for (w, idx) in zip(vocab.keys(), vocab.values()) }

for model in (clf, linear_clf):
	print '\nTop predictors of spam: '
	sorted_indices = [i for i in 
                        reversed( np.argsort(model.coef_.flatten()) )]
	for i in range(20):
		idx = sorted_indices[i]
		print(' %-15s (%f)' % ( new_vocab[idx+1], clf.coef_[:,idx] ) )
	print '\n'

raw_input('<Press Enter to continue>')

# =================== Part 6: Try Your Own Emails =====================
#  Now that you've trained the spam classifier, you can use it on your own
#  emails! In the starter code, we have included spamSample1.txt,
#  spamSample2.txt, emailSample1.txt and emailSample2.txt as examples. 
#  The following code reads in one of these emails and then uses your 
#  learned SVM classifier to determine whether the email is Spam or 
#  Not Spam
files = ['emailSample1.txt', 'emailSample2.txt', 
         'spamSample1.txt', 'spamSample2.txt', 
         'spamSample3.txt', 'spamSample4.txt']

for filename in files:
	word_indices, words = process_email(filename)
	x = emailFeatures(word_indices)
	p = clf.predict(x)
	print('\nProcessed %s' % filename);
	# printEmail(words)
	print('Spam Classification: %d' % p[0])
	print('(1 indicates spam, 0 indicates not spam)');






