from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import json
import collections

import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec

import config


class ALSTPData(object):
	def __init__(self, num_steps, is_training):
		""" For each epoch, the correct order for the dataset
		    should be shuffle, sample, next user, next item.
		"""
		self.num_steps = num_steps
		self.is_training = is_training

		self.doc2vec = Doc2Vec.load(config.doc2model_path)
		query_dict = json.load(open(config.query_path, 'r'))

		split = config.train_path if self.is_training else config.test_path
		self.data = pd.read_csv(split, 
						usecols=['query_', 'reviewerID', 'asin'])
		self.user_list = self.data.reviewerID.unique()
		self.item_list = self.data.asin.unique()

		# match asin and query vectors
		item_vec = []
		query_vec = []
		for i in range(len(self.data)):
			item_vec.append(
					self.doc2vec.docvecs[self.data.asin[i]])
			query_vec.append(self.doc2vec.docvecs[
					query_dict[self.data.query_[i]]])

		self.data['item_vec'] = item_vec
		self.data['query_vec'] = query_vec

	@property
	def _get_u_num(self):
		return len(self.user_list)

	@property
	def _get_i_num(self):
		return len(self.data.asin.unique())

	def shuffle_user(self):
		np.random.shuffle(self.user_list)
		return self.user_list

	def neg_sample(self, neg_num=5):
		neg_vecs = []
		for i in range(len(self.data)):
			item = self.data.asin[i]
			negs = np.random.choice(self.item_list, neg_num, False)
			while item in negs:
				negs = np.random.choice(self.item_list, neg_num, False)
			negs = [self.doc2vec.docvecs[k] for k in negs]
			neg_vecs.append(np.array(negs))
		self.data['neg_vecs'] = neg_vecs

	def next_user(self, user):
		userDF = self.data[
				self.data.reviewerID == user].reset_index(drop=True)
		self.item_u = np.array(userDF.asin)
		self.item_vec_u = np.array(userDF.item_vec.tolist())
		self.query_vec_u = np.array(userDF.query_vec.tolist())
		self.text_query_u = userDF.query_.tolist()

		# return specific user training items number
		if self.is_training:
			self.neg_vecs_u = np.array(userDF.neg_vecs.tolist())
			self.u_len = len(userDF) - self.num_steps
			return self.u_len

	def next_item(self, asin):
		"""asin is the user's next tuple, when a user ends, reset asin to 0.
		   This must be called after methods shuffle and next user.
		"""
		if self.is_training:
			# save the last user and query_vec_u
			if asin == self.u_len:
				return (self.item_vec_u[asin: asin + self.num_steps],
					    self.item_u[asin: asin + self.num_steps],
					    self.query_vec_u[asin: asin + self.num_steps])
			else:
				return  (self.item_vec_u[asin: asin + self.num_steps],
						 self.item_vec_u[asin + self.num_steps],
						 self.neg_vecs_u[asin + self.num_steps],
						 self.item_u[asin + self.num_steps],
					     self.query_vec_u[asin: asin + self.num_steps],
					     self.query_vec_u[asin + self.num_steps])

		else:
			return (self.item_u[0], 
					self.query_vec_u, 
					self.text_query_u[0])
