from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class ALSTP(object):
	def __init__(self, fix_dim, num_steps, global_dim, clip_norm,
				lr, activation_func, optim, num_negative, alpha,
				dropout, item_num, regularizer_rate, topk, is_training):
		"""
		Important Args.
			clip_norm: the global clipping norm rate.
			fix_dim: pre-defined dimension from doc2vec.
			num_negative: sampled negative item numbers.
			num_steps: the number of previous purchased products.
			alpha: the long-term preference updating rate.
			item_num: the number of all items.
			regularizer_rate: L2 is chosen.
		"""
		self.lr = lr
		self.activation_func = activation_func
		self.optim = optim
		self.clip_norm = clip_norm
		self.fix_dim = fix_dim
		self.num_steps = num_steps
		self.global_dim = global_dim
		self.alpha = alpha
		self.dropout = dropout
		self.num_negative = num_negative
		self.item_num = item_num
		self.regularizer_rate = regularizer_rate
		self.topk = topk
		self.is_training = is_training

		self.lr = tf.get_variable(name='lr', shape=[], 
						trainable=False,
						initializer=tf.constant_initializer(lr))

		self.local_dim = int(0.4 * global_dim)
		self.batch_size = 1 #Without parallism

	def inference(self):
		""" Initialize important settings. """
		self.regularizer = tf.contrib.layers.l2_regularizer(
							self.regularizer_rate)

		if self.activation_func == 'ReLU':
			self.activation_func = tf.nn.relu 
		elif self.activation_func == 'Leaky_ReLU':
			self.activation_func = tf.nn.leaky_relu 
		elif self.activation_func == 'ELU':
			self.activation_func = tf.nn.elu 

		if self.optim == 'SGD':
			self.optim = tf.train.GradientDescentOptimizer(
							self.lr, name='SGD')
		elif self.optim == 'Momentum':
			self.optim = tf.train.MomentumOptimizer(self.lr, 
							momentum=0.9, name='RMSProp')
		elif self.optim == 'Adam':
			self.optim = tf.train.AdamOptimizer(self.lr, name='Adam')

		self.global_interest = tf.get_variable("global_interest",
	    				shape=[self.global_dim, 1],
	    				initializer=tf.zeros_initializer,
	    				trainable=False)

		self.gru = tf.contrib.rnn.GRUCell(self.global_dim, name='gru')

		with tf.name_scope("input"):
		    self.item_pre = tf.placeholder(dtype=tf.float32,
		    					name='item_pre',
		    					shape=[self.num_steps, self.fix_dim])
		    self.item_target = tf.placeholder(dtype=tf.float32,
		    					name='item_target',
		    					shape=[1, self.fix_dim])
		    self.query_pre =  tf.placeholder(dtype=tf.float32,
		    					name='query_pre',
		    					shape=[self.num_steps, self.fix_dim])
		    self.query_target =  tf.placeholder(dtype=tf.float32,
		    					name='query_target',
		    					shape=[1, self.fix_dim])
		    self.neg_samples = tf.placeholder(dtype=tf.float32,
		    					name='neg_samples',
		    					shape=[self.num_negative, self.fix_dim])
		    self.all_items = tf.placeholder(dtype=tf.float32,
		    					name='all_items',
		    					shape=[self.item_num, self.fix_dim])

	def create_model(self):
		""" Create model from scratch. """
		with tf.name_scope("convert"):
			self.item_convert_pre = tf.layers.dense(self.item_pre, 
							units=self.global_dim,
							activation=self.activation_func,
							kernel_regularizer=self.regularizer,
							name='convert_matrix',
							reuse=tf.AUTO_REUSE)
			self.item_convert_target = tf.layers.dense(self.item_target, 
							units=self.global_dim,
							activation=self.activation_func,
							kernel_regularizer=self.regularizer,
							name='convert_matrix',
							reuse=tf.AUTO_REUSE)
			self.query_convert_pre = tf.layers.dense(self.query_pre, 
							units=self.global_dim,
							activation=self.activation_func,
							kernel_regularizer=self.regularizer,
							name='convert_matrix',
							reuse=tf.AUTO_REUSE)
			self.query_convert_target = tf.layers.dense(self.query_target, 
							units=self.global_dim,
							activation=self.activation_func,
							kernel_regularizer=self.regularizer,
							name='convert_matrix',
							reuse=tf.AUTO_REUSE)
			self.neg_convert_samples = tf.layers.dense(self.neg_samples, 
							units=self.global_dim,
							activation=self.activation_func,
							kernel_regularizer=self.regularizer,
							name='convert_matrix',
							reuse=tf.AUTO_REUSE)
			self.all_convert_items = tf.layers.dense(self.all_items, 
							units=self.global_dim,
							activation=self.activation_func,
							kernel_regularizer=self.regularizer,
							name='convert_matrix',
							reuse=tf.AUTO_REUSE)

		with tf.name_scope("local_context"):
			self.initial_state = tf.transpose(
						self.global_interest, name='init_gru_state')
			self.outputs, self.state = tf.nn.dynamic_rnn(cell=self.gru,
							inputs=tf.reshape(self.item_convert_pre, 
							[1, self.num_steps, self.global_dim]),
							initial_state=self.initial_state,
							dtype=tf.float32)
			self.outputs_reshape = tf.reshape(self.outputs, 
							[self.num_steps, self.global_dim])

			# Compute local attention weights for items.
			self.queries_local = tf.layers.dense(self.query_convert_pre,
							units=self.local_dim,
							activation=None,
							kernel_regularizer=self.regularizer,
							name='queries_local')
			self.query_local = tf.layers.dense(self.query_convert_target,
							units=self.local_dim,
							activation=None,
							kernel_regularizer=self.regularizer,
							name='query_local')
			self.v_local = tf.layers.dense(
							self.activation_func(
							self.query_local+self.queries_local),
							units=1,
							activation=None,
							kernel_regularizer=self.regularizer,
							name='v_local')
			assert self.v_local.shape == [self.num_steps, 1]

			self.local_weights = tf.nn.softmax(self.v_local, axis=0,
							name='local_weights')

			self.local_context = tf.reduce_sum(tf.multiply(
							self.outputs_reshape, self.local_weights),
							axis=0, keepdims=True, name='local_context')

		with tf.name_scope("global_context"):
			# Compute global attention weights for global interest.
			self.query_global = tf.layers.dense(self.query_target,
							units=1,
							activation=self.activation_func,
							kernel_regularizer=self.regularizer,
							name='query_global')
			self.global_new = tf.matmul(self.global_interest, 
							self.query_global, name='global_new')
			self.global_weights = tf.nn.softmax(self.global_new, axis=0,
							name='global_weights')

			self.global_context = tf.transpose(tf.multiply(
							self.global_interest, 
							self.global_weights),
							name='global_context')

		with tf.name_scope("concatenation"):
			self.concate = tf.concat([self.local_context, self.global_context,
							self.query_convert_target], axis=1)

			self.query_integration = tf.layers.dense(self.concate,
							units=self.global_dim,
							activation=self.activation_func,
							kernel_regularizer=self.regularizer,
							name='query_integration')

		with tf.name_scope("similarity"):
			self.target_item = tf.nn.l2_normalize(
							self.item_convert_target, 1)
			self.samples_neg = tf.nn.l2_normalize(
							self.neg_convert_samples, 1)
			self.query_integration = tf.nn.l2_normalize(
							self.query_integration, 1)
			self.items_all = tf.nn.l2_normalize(
							self.all_convert_items, 1)

			self.query_integration_negs = tf.concat(
						[self.query_integration for _ in 
						range(self.num_negative)], axis=0)
			self.query_integration_all = tf.concat(
						[self.query_integration for _ in
						range(self.item_num)], axis=0)

			self.pos_score = 1 - tf.losses.cosine_distance(
							self.target_item, self.query_integration, 
							reduction=tf.losses.Reduction.NONE, axis=1)
			self.neg_scores = 1 - tf.losses.cosine_distance(
							self.samples_neg, self.query_integration_negs, 
							reduction=tf.losses.Reduction.NONE, axis=1)
			self.items_all_scores = 1- tf.losses.cosine_distance(
							self.items_all, self.query_integration_all,
							reduction=tf.losses.Reduction.NONE, axis=1)

		with tf.name_scope("loss"):
			reg = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
			self.reg_loss = tf.contrib.layers.apply_regularization(
							self.regularizer, reg)

			self.pos_score = tf.reshape(self.pos_score, [-1])
			self.neg_scores = tf.reshape(self.neg_scores, [-1])
			self.loss = -tf.reduce_mean(tf.log(tf.sigmoid(
						self.pos_score-self.neg_scores))) + self.reg_loss

		with tf.name_scope("optimizer"):
			tvars = tf.trainable_variables()
			grads, _ = tf.clip_by_global_norm(tf.gradients(
								self.loss, tvars), self.clip_norm)
			self.optimizer = self.optim.apply_gradients(zip(grads, tvars))

		self.new_lr = tf.placeholder(tf.float32, shape=[], name='new_lr')
		self.lr_update = tf.assign(self.lr, self.new_lr)

	def assign_lr(self, sess, lr_value):
		sess.run(self.lr_update, feed_dict={self.new_lr: lr_value})

	def set_global_interest(self):
		""" For initializing, constucting and updating global interest
			based on specific conditions.
		"""
		self.init_global_interest = tf.assign(self.global_interest, 
			tf.zeros([self.global_dim, 1]), name='init_global_interest')
		
		self.old_state = tf.placeholder(tf.float32,
						shape=[1, self.global_dim], name='old_state')
		self.state_trans = tf.transpose(self.old_state)
		self.global_interest_con = tf.assign(self.global_interest,
					self.state_trans, name='global_interest_con')

		self.global_interest_update = tf.assign(self.global_interest,
							self.global_interest.value()*self.alpha + 
							self.state_trans*(1-self.alpha), 
							name='global_interest_update')

		self.new_global_interest = tf.placeholder(tf.float32, 
			shape=[self.global_dim, 1], name='new_global_interest')
		self.global_interest_assign = tf.assign(
					self.global_interest, self.new_global_interest)

	def cons_global_interest(self, sess, old_state):
		sess.run(self.global_interest_con, feed_dict={
							self.old_state: old_state})

	def update_global_interest(self, sess, old_state):
		sess.run(self.global_interest_update, feed_dict={
							self.old_state: old_state})

	def assign_global_interest(self, sess, global_interest):
		sess.run(self.global_interest_assign, feed_dict={
					self.new_global_interest: global_interest})

	def eval(self):
		with tf.name_scope("evaluation"):
			self.items_all_scores = tf.reshape(self.items_all_scores, [-1])

			_, self.items_topk_indices = tf.nn.top_k(
					self.items_all_scores, k=self.topk, name='top_k')

	def summary(self):
		""" Create summaries to write on tensorboard. """
		self.writer = tf.summary.FileWriter('./graphs/ALSTP', 
								tf.get_default_graph())
		with tf.name_scope("summaries"):
			tf.summary.scalar('loss', self.loss)
			tf.summary.histogram('histogram loss', self.loss)
			self.summary_op = tf.summary.merge_all()

	def build(self):
		""" Build the computation graph. """
		self.inference()
		self.create_model()
		self.set_global_interest()
		self.eval()
		self.summary()
		self.saver = tf.train.Saver(tf.global_variables())

	def step(self, sess, item_pre, item_target, query_pre, 
					query_target, neg_samples, all_items, step):
		""" Start trainging the model. """
		input_feed = {}
		input_feed[self.item_pre.name] = item_pre
		input_feed[self.query_pre.name] = query_pre
		input_feed[self.query_target.name] = query_target


		if self.is_training:
			input_feed[self.item_target.name] = item_target
			input_feed[self.neg_samples.name] = neg_samples

			output_feed = [self.state, self.global_interest,
							self.loss, self.optimizer, 
							self.lr, self.summary_op]
			outputs = sess.run(output_feed, input_feed)
			self.writer.add_summary(outputs[-1], global_step=step)

		else:
			input_feed[self.all_items.name] = all_items

			output_feed = [self.items_topk_indices]
			outputs = sess.run(output_feed, input_feed)

		return outputs
