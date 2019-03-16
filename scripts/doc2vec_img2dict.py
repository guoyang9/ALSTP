import os, argparse
import random, json
import collections

import numpy as np
import pandas as pd

from ast import literal_eval
from gensim.models import doc2vec


parser = argparse.ArgumentParser()

parser.add_argument("--dataset", default='MenClothing', type=str,
			help="choose dataset to process.")
parser.add_argument("--embed_size", default=512, type=int,
			help="doc dimension.")
parser.add_argument("--window_size", default=3, type=int,
			help="sentence window size.")

FLAGS = parser.parse_args()

ROOT_DIR = '../processed/'

full_file = os.path.join(ROOT_DIR, FLAGS.dataset, 'full.csv')
full_data = pd.read_csv(full_file, usecols=[
								'query_', 'reviewText', 'asin'])
full_data.query_ = full_data.query_.apply(literal_eval)
full_data.reviewText = full_data.reviewText.apply(literal_eval)

#Gather reviews to same asins.
rawDoc = collections.defaultdict(list)
for k, v in zip(full_data.asin, full_data.reviewText):
	rawDoc[k].append(v)

#Concatenate the reviews together.
for k in rawDoc.keys():
	m = []
	for i in rawDoc[k]:
		m.extend(i)
	rawDoc[k] = m

#For query, it's hard to tag, so we just random tag them.
query_idx = 0
query_dic = {} #For query index and doc2vec index matching.
for q in full_data['query_']:
	if repr(q) not in query_dic:
			query_dic[repr(q)] = query_idx
			rawDoc[query_idx] = q * 20
			query_idx += 1

##################################### Start Model ################################
docs = []
analyzedDocument = collections.namedtuple('AnalyzedDocument', 'words tags')
for d in rawDoc.keys():
	docs.append(analyzedDocument(rawDoc[d], [d]))

alpha_val = 0.025
min_alpha_val = 1e-4
passes = 20

alpha_delta = (alpha_val - min_alpha_val) / (passes - 1)

model = doc2vec.Doc2Vec(vector_size=FLAGS.embed_size,
			window=FLAGS.window_size, min_count=2, workers=6, epochs=20)

model.build_vocab(docs) # Building vocabulary

for epoch in range(passes):
    random.shuffle(docs)

    model.alpha, model.min_alpha = alpha_val, alpha_val
    model.train(docs, total_examples=len(docs), epochs=model.iter)

    alpha_val -= alpha_delta

#Save model
model.save(os.path.join(ROOT_DIR, FLAGS.dataset, 'doc2vecFile'))

#Write query embedding and index to disk
json.dump(query_dic, open(os.path.join(
						ROOT_DIR, FLAGS.dataset, 'queryFile.json'), 'w'))
print("All finished!!!\nThe query number is %d." %len(query_dic))
