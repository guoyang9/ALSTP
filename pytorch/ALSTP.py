from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import time 
import argparse
import numpy as np
sys.path.append(os.getcwd())
from gensim.models.doc2vec import Doc2Vec

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter

import config
import evaluate
import data_input


class ALSTP(nn.Module):
	def __init__(self, fix_dim, num_steps, global_dim, 
									alpha, dropout, is_training):
		super(ALSTP, self).__init__()
		""" 
		Important Args.
		fix_dim: pre-defined dimension from doc2vec.
		num_steps: the number of previous purchased products.
		alpha: the long-term preference updating rate.
		dropout: drop rate.
		"""
		self.fix_dim = fix_dim
		self.num_steps = num_steps
		self.global_dim = global_dim
		self.alpha = alpha
		self.dropout = dropout
		self.is_training = is_training

		self.local_dim = int(0.4 * global_dim)
		self.batch_size = 1 # without parallism

		self.global_interest = torch.zeros(self.global_dim, 1).cuda()

		self.gru = nn.GRU(self.global_dim, self.global_dim, 1)
		for param in self.gru.parameters():
			if param.dim() == 2:
				nn.init.xavier_uniform_(param)
			else:
				param.data.fill_(0)

		# convert items and queries
		self.convert = nn.Linear(self.fix_dim, self.global_dim)

		# local Attention part
		self.queries_local = nn.Linear(self.global_dim, self.local_dim)
		self.query_local = nn.Linear(self.global_dim, self.local_dim)
		self.v_local = nn.Linear(self.local_dim, 1)
	

		# global Attention part
		self.query_global = nn.Linear(self.global_dim, 1)

		# concatenation weights and bias
		self.concate = nn.Sequential(
			nn.Linear(3 * self.global_dim, 1 * self.global_dim),
			# nn.ELU(),
			# nn.Dropout(p=dropout),
			# nn.Linear(2 * self.global_dim, 2 * self.global_dim),
			# nn.ELU(),
			# nn.Dropout(p=dropout),
			# nn.Linear(2 * self.global_dim, self.global_dim),
			nn.ELU()
		)

		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight)
				if m.bias is not None:
					m.bias.data.zero_()

	def init_global_interest(self):
		""" When a new user starts, the global interest should be
			initialized to zeros.
		"""
		self.global_interest.fill_(0)

	def cons_global_interest(self):
		final_state = self.state[-1].data.view(self.global_dim, 1)
		self.global_interest = final_state

	def update_global_interest(self):
		""" final_state is the rnn final hidden state,
			after a num_steps rounds, update.
		"""
		final_state = self.state[-1].data.view(self.global_dim, 1)
		self.global_interest = self.alpha * self.global_interest + (
											1-self.alpha) * final_state

	def forward(self, item_pre, item_target, query_pre, query_target,
												neg_samples, all_items):
		# convert items and queies to the same space
		item_pre = F.elu(self.convert(item_pre))
		query_pre = F.elu(self.convert(query_pre))
		query_target = F.elu(self.convert(query_target))

		# for computing local context
		self.state = self.global_interest.view(
								1, self.batch_size, self.global_dim)
		hidden, self.state = self.gru(
						item_pre.view(self.num_steps, 
						self.batch_size, self.global_dim), self.state)
		hidden = hidden.view(self.num_steps, self.global_dim)

		local_weights = F.softmax(self.v_local(F.elu(
							self.queries_local(query_pre) + 
							self.query_local(query_target))), dim=0)

		localContext = torch.sum(hidden*local_weights, 0).view(1, self.global_dim)

		# compute global context
		_global_weights = F.softmax(torch.mm(
							self.global_interest, F.elu(
							self.query_global(query_target))), dim=0)
		globalContext =  (self.global_interest * _global_weights
												).view(1, self.global_dim)

		# concatenation part
		concate = torch.cat([localContext, globalContext, query_target], 1)
		concate_query = self.concate(concate)

		if self.is_training:
			item_target = F.elu(self.convert(item_target))
			neg_samples = F.elu(self.convert(neg_samples))

			pos_score = F.cosine_similarity(item_target, concate_query)
			neg_scores = F.cosine_similarity(neg_samples, concate_query)
			return pos_score, neg_scores
		else:
			all_items = F.elu(self.convert(all_items))
			all_scores = F.cosine_similarity(all_items, concate_query)
			return all_scores


def BPRLoss(pos_score, neg_scores):
		return -torch.mean(torch.log(
							torch.sigmoid(pos_score - neg_scores)), 0)


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("--dataset", default='Clothing', type=str,
		help="the chosen dataset.")
	parser.add_argument("--lr", 
		type=float,
		default=0.001, 
		help="leaning rate")
	parser.add_argument("--clip_norm", 
		type=float,
		default=5.0, 
		help="global clip norm")
	parser.add_argument("--alpha", 
		type=float,
		default=0.9, 
		help="long-term preference update rate")
	parser.add_argument("--dropout", 
		type=float,
		default=0.0, 
		help="drop out rate")
	parser.add_argument("--weight_decay", 
		type=float,
		default=0.0001, 
		help="weight decay rate")
	parser.add_argument("--train_epoch", 
		type=int,
		default=20, 
		help="train epoch number")
	parser.add_argument("--num_steps", 
		type=int,
		default=4, 
		help="num_steps in GRU and update rounds")
	parser.add_argument("--global_dim", 
		type=int,
		default=128, 
		help="the global dimension for items and queries")
	parser.add_argument("--negative_numbers", 
		type=int,
		default=5, 
		help="sample negtive numbers")
	parser.add_argument("--model_dir", 
		type=str,
		default="/model_tmp", 
		help="directory for model saving")
	parser.add_argument("--top_k", 
		type=int,
		default=20, 
		help="return the top k results")
	parser.add_argument("--gpu", 
		type=str,
		default="0", 
		help="gpu card ID")
	FLAGS = parser.parse_args()

	opt_gpu = FLAGS.gpu
	os.environ["CUDA_VISIBLE_DEVICES"] = opt_gpu

	############################### CREATE MODEL ###########################
	model = ALSTP(config.embed_size, FLAGS.num_steps, FLAGS.global_dim,
						FLAGS.alpha, FLAGS.dropout,  is_training=True)
	model.cuda()
	optimizer = optim.SGD(model.parameters(), lr=FLAGS.lr, 
						momentum=0.9, weight_decay=FLAGS.weight_decay)
	scheduler = optim.lr_scheduler.StepLR(
									optimizer, step_size=5, gamma=0.5)

	############################### PREPARE DATA ############################
	train_data = data_input.ALSTPData(FLAGS.num_steps, is_training=True)
	valid_data = data_input.ALSTPData(None, is_training=False)
	# for testing
	full_data = data_input.ALSTPData(None, is_training=False)
	all_items, all_item_vecs = [], []
	for i in range(len(full_data.data)):
		item = full_data.data['asin'][i]
		if not item in all_items:
			all_items.append(item)
			all_item_vecs.append(full_data.data['item_vec'][i])

	writer = SummaryWriter() # for visualizing loss
	writer_count = 0
	
	############################# START TRAINING #############################
	for epoch in range(FLAGS.train_epoch):
		train_data.neg_sample(FLAGS.negative_numbers)
		model.train() 
		model.is_training = True
		start_time = time.time()
		(global_int_eval, item_pre_eval, query_vec_eval, 
				item_id_eval) = ([] for _ in range(4)) # for evaluation

		user_list = train_data.shuffle_user() # first shuffle all users
		
		for step, user in enumerate(user_list):
			model.init_global_interest() # initializing global interest to zeros
			item_len = train_data.next_user(user)
			for itemidx in range(item_len):
				(item_pre, item_target, neg_vecs, target_id, query_pre,
								query_target) = train_data.next_item(itemidx)

				item_pre = torch.tensor(item_pre).cuda()
				query_pre = torch.tensor(query_pre).cuda()
				item_target = torch.tensor(
							item_target).cuda().view(1, config.embed_size)             
				query_target = torch.tensor(
							query_target).cuda().view(1, config.embed_size)
				negative_samples = torch.tensor(neg_vecs).cuda()

				if itemidx == 1:
					model.cons_global_interest()

				# slowly update global interest
				if not itemidx == 0 and itemidx % FLAGS.num_steps == 0:
					model.update_global_interest()

				model.zero_grad()
				pos_score, neg_scores = model(item_pre, item_target,
												query_pre, query_target, 
												negative_samples, None)
				loss = BPRLoss(pos_score, neg_scores)
				loss.backward()
				nn.utils.clip_grad_norm_(model.parameters(), FLAGS.clip_norm)
				optimizer.step()

				writer.add_scalar('data/loss', loss.data.item(), writer_count)
				writer_count += 1

	########################## SAVE FOR EVALUATION ##########################
			item_pre, itemID, query_pre = train_data.next_item(item_len)
			item_pre_eval.append(torch.tensor(item_pre))
			query_vec_eval.append(torch.tensor(query_pre))
			global_int_eval.append(model.global_interest)
			item_id_eval.append(itemID)


		evaluate.valid(model, valid_data, config.embed_size, 
						user_list, global_int_eval, item_pre_eval, 
						query_vec_eval, item_id_eval, all_items, 
						all_item_vecs, FLAGS.top_k)
		elapsed_time = time.time() - start_time
		scheduler.step(epoch)
		print("Epoch: {:d} time is:\t".format(epoch)
						+ time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))


if __name__ == "__main__":
	main()
