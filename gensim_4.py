# -*- coding: utf-8 -*-
"""
Installation:
sudo pip3 install gensim
sudo pip3 install nltk
---
import nltk
nltk.download('punkt')
---
git clone https://github.com/mhq/train_punkt.git
---
USE >>nltk.download()
to download texts and models.
==================

TUTORIAL:
https://www.oreilly.com/learning/how-do-i-compare-document-similarity-using-python

"""
# imports
import numpy as np
import gensim
#import string
#LabeledSentence = gensim.models.doc2vec.LabeledSentence
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize

#from nltk.tokenize import sent_tokenize
#from nltk.corpus import stopwords
#from nltk.tokenize import word_tokenize
import os
import os.path

import database_get_docs

DEBUG = False


def load_docs_from_dir(data_path):

	#data_path = 'texts'
	docLabels = [f for f in os.listdir(data_path) if f.endswith('.html')]
	print('num of texts:', len(docLabels))

	data = []
	for i, doc in enumerate(docLabels):
		with open(data_path + '/' + doc, 'r') as f:
			text = f.read()
			text = text.replace('\n', ' ')
			print('{}: {}, len={}'.format(i, doc, len(text)))
			data.append(text)

	return data

def get_example_data():

	raw_docs = ["Зеленый чайник на кровати.",
		"Мой компьютер и я",
		"Не знаю что написать.",
		"Море завет меня, и ты",
		"Смешно, да, опять."]
	print("Number of documents:",len(raw_docs))
	return raw_docs


def create_model(docs):

	#raw_docs = load_docs(data_path='texts')
	#raw_docs = get_example_data()
	
	raw_docs = docs['texts']

	print('Text example:')
	print('Doc-0:', raw_docs[0][:500])
	print('Doc-1:', raw_docs[1][:500])


	gen_docs = [[w.lower() for w in word_tokenize(text)] 
		for text in raw_docs]	
	print('size of gen_docs:', len(gen_docs))

	# print whole docs:
	#for i in range(len(gen_docs)):
	#	print('doc {}: {}', i, gen_docs[i])

	dictionary = gensim.corpora.Dictionary(gen_docs)
	print('The number of words in the dictionary:', len(dictionary))
	print(dictionary[0])
	#print(dictionary[5])
	print('я -',  dictionary.token2id['я'])
	#print('ты -', dictionary.token2id['ты'])
	print('синдром -', dictionary.token2id['синдром'])
	print("Number of words in dictionary:",len(dictionary))
	
	# output WHOLE DICTIONARY:
	#for i in range(len(dictionary)):
	#	print(i, dictionary[i])

	# Create a corpus. A corpus is a list of bags of words.	
	# A bag-of-words representation for a document just lists 
	# the number of times each word occurs in the document.	
	corpus = [dictionary.doc2bow(gen_doc) for gen_doc in gen_docs]
	if DEBUG: print(corpus)	# [[(0, 1), (1, 1), (2, 1), (3, 2), ...

	# Create a tf-idf model from the corpus
	tf_idf_model = gensim.models.TfidfModel(corpus)
	if DEBUG: print(tf_idf_model) # TfidfModel(num_docs=5, num_nnz=47)
	# num_nnz is the number of tokens.
	s = 0
	for i in corpus:
		s += len(i)
	print('s =', s)

	#  create a similarity measure object in tf-idf space.
	sims = gensim.similarities.Similarity(
		'./workdir/', tf_idf_model[corpus],
		num_features=len(dictionary))
	print(sims)
	print(type(sims))	

	# create a query document and convert it to tf-idf.
	query_doc = [w.lower() for w in word_tokenize(\
		"у меня амнезия")]

	print(query_doc)
	query_doc_bow = dictionary.doc2bow(query_doc)
	#print(query_doc_bow)
	query_doc_tf_idf_model = tf_idf_model[query_doc_bow]
	print('result:', query_doc_tf_idf_model)

	sim = sims[query_doc_tf_idf_model]
	for i, s in enumerate(sim):
		print(i, ': s =', s)

	#sim_arr = np.array(sim)
	max_idx = np.argmax(sim)	
	print('max idx =', max_idx)
	print('max val =', sim[max_idx])
	print(max(sim))


	max_idxs = (-sim).argsort()[:10]
	for idx in max_idxs:
		print('{:.4f} - {:3d}: id={}, name={}'.format(
			sim[idx], idx, docs['ids'][idx], docs['names'][idx]))

	return max_idxs

	"""
	print('Tokenize')
	tagged_data = [TaggedDocument(words=word_tokenize(d.lower()), \
		tags=[str(i)]) for i, d in enumerate(data)]

	#max_epochs = 100
	#vec_size = 20
	#alpha = 0.025
	
	max_epochs = 100
	vec_size = 20
	alpha = 0.025

	print('Doc2Vec')
	model = Doc2Vec(#size=vec_size,
					vector_size=vec_size,	
					alpha=alpha, 
					#min_alpha=0.00025,
					min_alpha=0.025,
					min_count=1,
					dm=1,
					workers=3)

	print('build_vocab')
	model.build_vocab(tagged_data)

	for epoch in range(max_epochs):
		print('iteration {0}'.format(epoch))
		model.train(tagged_data,
				total_examples=model.corpus_count,
				epochs=model.iter)
		# decrease the learning rate
		model.alpha -= 0.0002
		# fix the learning rate, no decay
		model.min_alpha = model.alpha


	return model
	"""

if __name__ == '__main__':

	docs = database_get_docs.load_docs_from_db()
	max_idxs = create_model(docs)

	#model.save("d2v.model")
	#print("Model Saved")
