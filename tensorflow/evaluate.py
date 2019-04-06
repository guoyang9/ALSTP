import os, sys

import numpy as np 
import tensorflow as tf


def mrr(gt_item, pred_items):
	if gt_item in pred_items:
		index = np.where(pred_items == (gt_item))[0][0]
		return np.reciprocal(float(index+1))
	else:
		return 0


def hit(gt_item, pred_items):
	if gt_item in pred_items:
		return 1
	return 0


def ndcg(gt_item, pred_items):
	if gt_item in pred_items:
		index =np.where(pred_items == (gt_item))[0][0]
		return np.reciprocal(np.log2(index+2))
	return 0


def valid(model, sess, valid_data, fix_dim, user_list, 
			global_int_eval, item_vec_eval,	query_vec_eval, 
			item_id_eval, all_items, all_items_vec):
	"""
	Import Args:
	user_list: correct user sequential list.
	global_int_eval: corresponding to each user's global interest.
	item_vec_eval: the ones before the ground truth one.
	query_vec_eval: the queries before the target one.
	item_id_eval: the previous item ids.
	all_items: all item ids.
	all_items_vec: all items' vector representation.
	"""
	HR, MRR, NDCG = [], [], []

	# inverse list and pop them one by one
	global_int_eval = global_int_eval[::-1]
	item_vec_eval = item_vec_eval[::-1]
	query_vec_eval = query_vec_eval[::-1]
	item_id_eval = item_id_eval[::-1]

	model.is_training = False

	for user in user_list:
		global_interest = np.reshape(global_int_eval.pop(), [-1, 1])
		model.assign_global_interest(sess, global_interest)
		item_vec = item_vec_eval.pop()
		query_pre = query_vec_eval.pop()
		item_ids = item_id_eval.pop()

		_ = valid_data.next_user(user)
		item_id, query_target, text_query = valid_data.next_item(0)
		query_target = np.reshape(query_target, [1, fix_dim])

		items_indices = model.step(sess, item_vec, None, query_pre, 
							query_target, None, all_items_vec, None)
		items_rank = np.take(all_items, items_indices[0])

		HR.append(hit(item_id, items_rank))
		MRR.append(mrr(item_id, items_rank))
		NDCG.append(ndcg(item_id, items_rank))

	print("Hit rate is %.3f\t Mrr is %.3f\t Ndcg is %.3f" % (
												np.mean(HR), 
												np.mean(MRR),
												np.mean(NDCG)))
