from __future__ import absolute_import
from __future__ import division
from __future__ import print_functionafrom __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import numpy as np

import torch
import torch.nn.functional as F

import data_import


def mrr(gt_item, pred_items):
	if gt_item in pred_items:
		index = torch.tensor(pred_items.tolist().index(gt_item), 
								dtype=torch.float32)
		return torch.reciprocal(index+1)
	else:
		return 0

def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0

def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = torch.tensor(pred_items.tolist().index(gt_item), 
								dtype=torch.float32)
		return torch.reciprocal(torch.log2(index+2))
	return 0


def valid(model, valid_data, fix_dim, user_list, 
			global_int_eval, item_vec_eval,
			query_vec_eval, item_id_eval, 
			all_items, all_items_vec, top_k):
	model.eval() #Turn off dropout
	HR, MRR, NDCG = [], [], []

	#Inverse list and pop them one by one
	global_int_eval = global_int_eval[::-1]
	item_vec_eval = item_vec_eval[::-1]
	query_vec_eval = query_vec_eval[::-1]
	item_id_eval = item_id_eval[::-1]

	all_items_vec = torch.tensor(
				torch.cuda.FloatTensor(np.array(all_items_vec)))

	model.is_training = False

	for user in user_list:
		model.global_interest = global_int_eval.pop()
		item_vec = item_vec_eval.pop()
		query_pre = query_vec_eval.pop()
		item_ids = item_id_eval.pop()

		_ = valid_data.next_user(user)
		item_id, query_target, text_query = valid_data.next_item(0)
		query_target = torch.tensor(torch.cuda.FloatTensor(
							np.reshape(query_target, [1, fix_dim])))
		all_score = model(item_vec, None, query_pre, 
								query_target, None, all_items_vec)
		_, top_rank = torch.topk(all_score, top_k)
		items_rank = np.take(all_items, top_rank)

		top_rank = top_rank.data.cpu().numpy()
		HR.append(hit(item_id, items_rank))
		MRR.append(mrr(item_id, items_rank))
		NDCG.append(ndcg(item_id, items_rank))

	print("Hit rate is %.3f\t Mrr is %.3f\t Ndcg is %.3f" % (
								np.mean(HR), 
								np.mean(MRR),
								np.mean(NDCG)))


import os, sys
import numpy as np
import torch
import torch.nn.functional as F

import data_import


def mrr(gt_item, pred_items):
	if gt_item in pred_items:
		index = torch.tensor(pred_items.tolist().index(gt_item), 
							dtype=torch.float32)
		return torch.reciprocal(index+1)
	else:
		return 0

def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0

def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index = torch.tensor(pred_items.tolist().index(gt_item), 
							dtype=torch.float32)
		return torch.reciprocal(torch.log2(index+2))
	return 0


def valid(model, valid_data, fix_dim, user_list, 
			global_int_eval, item_vec_eval,
			query_vec_eval, item_id_eval, 
			all_items, all_items_vec, top_k):
	model.eval() #Turn off dropout
	HR, MRR, NDCG = [], [], []

	#Inverse list and pop them one by one
	global_int_eval = global_int_eval[::-1]
	item_vec_eval = item_vec_eval[::-1]
	query_vec_eval = query_vec_eval[::-1]
	item_id_eval = item_id_eval[::-1]

	all_items_vec = torch.tensor(
				torch.cuda.FloatTensor(np.array(all_items_vec)))

	model.is_training = False

	for user in user_list:
		model.global_interest = global_int_eval.pop()
		item_vec = item_vec_eval.pop()
		query_pre = query_vec_eval.pop()
		item_ids = item_id_eval.pop()

		_ = valid_data.next_user(user)
		item_id, query_target, text_query = valid_data.next_item(0)
		query_target = torch.tensor(torch.cuda.FloatTensor(
					np.reshape(query_target, [1, fix_dim])))
		all_score = model(item_vec, None, query_pre, 
					query_target, None, all_items_vec)
		_, top_rank = torch.topk(all_score, top_k)
		items_rank = np.take(all_items, top_rank)

		top_rank = top_rank.data.cpu().numpy()
		HR.append(hit(item_id, items_rank))
		MRR.append(mrr(item_id, items_rank))
		NDCG.append(ndcg(item_id, items_rank))

	print("Hit rate is %.3f\t Mrr is %.3f\t Ndcg is %.3f" % (
							np.mean(HR), 
							np.mean(MRR),
							np.mean(NDCG)))
