from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
sys.path.append(os.getcwd())

import numpy as np
import tensorflow as tf

from model import ALSTP
import evaluate, data_input


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('negative_num', 5, 'number of negative samples.')
tf.app.flags.DEFINE_integer('global_dim', 128, 'the size for preference.')
tf.app.flags.DEFINE_integer('epochs', 20, 'the number of epochs.')
tf.app.flags.DEFINE_integer('topK', 20, 'topk for evaluation.')
tf.app.flags.DEFINE_integer('num_steps', 4, 'number of previous products.')
tf.app.flags.DEFINE_integer('fix_dim', 512, 'size of learned from doc2vec.')
tf.app.flags.DEFINE_string('dataset', 'Clothing', 'the used dataset.')
tf.app.flags.DEFINE_string('optim', 'Momentum', 'the optimization method.')
tf.app.flags.DEFINE_string('activation', 'ELU', 'the activation function.')
tf.app.flags.DEFINE_string('model_dir', './', 'the dir for saving model.')
tf.app.flags.DEFINE_string('gpu', '0', 'the gpu card number.')
tf.app.flags.DEFINE_float('regularizer', 0.0001, 'the regularizer rate.')
tf.app.flags.DEFINE_float('lr', 0.001, 'learning rate.')
tf.app.flags.DEFINE_float('dropout', 0.0, 'dropout rate.')
tf.app.flags.DEFINE_float('clip_norm', 5.0, 'for avoiding gradient exploding.')
tf.app.flags.DEFINE_float('alpha', 0.9, 'updating rate for the global interest.')

opt_gpu = FLAGS.gpu
os.environ["CUDA_VISIBLE_DEVICES"] = opt_gpu


def train(train_data, valid_data, num_items, all_items, all_items_id):
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True

	with tf.Session(config=config) as sess:

		############################### CREATE MODEL #############################
		model = ALSTP(FLAGS.fix_dim, FLAGS.num_steps, FLAGS.global_dim,
					FLAGS.clip_norm, FLAGS.lr, FLAGS.activation, FLAGS.optim,
					FLAGS.negative_num, FLAGS.alpha, FLAGS.dropout,
					num_items, FLAGS.regularizer, FLAGS.topK, is_training=True)
		model.build()

		ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
		if ckpt:
			print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
			model.saver.restore(sess, ckpt.model_checkpoint_path)
		else:
			print("Creating model with fresh parameters.")
			sess.run(tf.global_variables_initializer())
		
		############################### Training ####################################
		count = 0
		for epoch in range(FLAGS.epochs):
			model.is_training = True
			train_data.neg_sample(FLAGS.negative_num)
			start_time = time.time()
			(global_int_eval, item_pre_eval, query_vec_eval,
					item_id_eval) = ([] for _ in range(4)) # for evaluation

			user_list = train_data.shuffle_user() # first shuffle all users

			for step, user in enumerate(user_list):
				sess.run(model.init_global_interest) # for a new user
				item_len = train_data.next_user(user)
				for itemidx in range(item_len):
					(item_pre, item_target, neg_vecs, target_id, query_pre,
								query_target) = train_data.next_item(itemidx)

					item_target = np.reshape(item_target, [1, FLAGS.fix_dim])
					query_target = np.reshape(query_target, [1, FLAGS.fix_dim])

					if itemidx == 1:
						model.cons_global_interest(sess, state)

					# slowly update global interest
					if not itemidx == 0 and itemidx % FLAGS.num_steps == 0:
						model.update_global_interest(sess, state)

					outputs = model.step(sess, item_pre, item_target, query_pre,
											query_target, neg_vecs, None, count)
					state = outputs[0] # for updating global_interest
					count += 1

		############################# SAVE FOR EVALUATION ##########################
				item_pre, itemID, query_pre = train_data.next_item(item_len)
				item_pre_eval.append(item_pre)
				query_vec_eval.append(query_pre)
				global_int_eval.append(outputs[1])
				item_id_eval.append(itemID)

			if (epoch+1) % 5 == 0:
				model.assign_lr(sess, outputs[4])
			evaluate.valid(model, sess, valid_data, FLAGS.fix_dim, user_list, 
							global_int_eval, item_pre_eval, query_vec_eval, 
							item_id_eval, all_items_id, all_items)

			elapsed_time = time.time() - start_time
			print("Epoch: %d\tEpoch Time is:\t" %(epoch)
					+ time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))

	    ################################## SAVE MODEL ################################
		checkpoint_path = os.path.join(FLAGS.model_dir, "NCF.ckpt")
		model.saver.save(sess, checkpoint_path)


def main(argv=None):
	train_data = data_input.ALSTPData(FLAGS.num_steps, is_training=True)
	valid_data = data_input.ALSTPData(None, is_training=False)

	# for testing
	full_data = data_input.ALSTPData(None, is_training=False)
	all_items_list, all_items_vec = [], []
	for i in range(len(full_data.data)):
		item = full_data.data['asin'][i]
		if not item in all_items_list:
			all_items_list.append(item)
			all_items_vec.append(full_data.data['item_vec'][i])

	num_items = len(all_items_list)
	all_items_id = np.array(all_items_list)
	all_items_vec = np.array(all_items_vec)

	train(train_data, valid_data, num_items, all_items_vec, all_items_id)
        

if __name__ == '__main__':
	tf.app.run()
