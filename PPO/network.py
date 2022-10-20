import numpy as np
import tensorflow_probability as tfp
import tensorflow as tf


class Actor():
	def __init__(self, args):
		super(Actor, self).__init__()
		self.eps = args.epsilon
		self.c_ent = args.entropy_weight
		self.l2 = args.l2

		self.get_model(args)

	def get_model(self, args):
		inputs = tf.keras.Input(shape=(3, )) # My position (0, 1), what I payed so far (0, 1) and how much time I spent in queue (0, 1)

		y = tf.keras.layers.Dense(args.hidden_layer_actor, activation='relu')(inputs)
		#y = tf.keras.layers.Dense(args.hidden_layer_size, activation='relu')(y)
		actions = tf.keras.layers.Dense(args.F + 1, activation='softmax')(y)

		self.model = tf.keras.Model(inputs=inputs, outputs=actions)
		self.model.compile(optimizer=tf.keras.optimizers.Adam(args.actor_learning_rate, global_clipnorm=args.clip_norm))

	@tf.function
	def train(self, x, advantage, actions, old_prob):
		with tf.GradientTape() as tape:
			prediction = self.model(x)
			dist = tfp.distributions.Categorical(probs=prediction)
			new_probs = dist.prob(actions)

			ratio = new_probs / old_prob

			clipped_advantage = tf.where(
				advantage > 0,
				(1 + self.eps) * advantage,
				(1 - self.eps) * advantage
			)

			policy_loss = - tf.reduce_mean(tf.minimum(ratio * advantage, clipped_advantage))

			entropy_loss = - self.c_ent * tf.reduce_mean(dist.entropy())

			l2_loss = 0
			for var in self.model.trainable_variables:
				l2_loss += self.l2 * tf.nn.l2_loss(var)

			loss = policy_loss + entropy_loss + l2_loss

		self.model.optimizer.minimize(loss, self.model.trainable_variables, tape=tape)

	@tf.function(experimental_relax_shapes=True)
	def predict(self, x):
		preds = self.model(x)
		dist = tfp.distributions.Categorical(probs=preds)
		actions = dist.sample()
		return actions, dist.prob(actions)


class Critic():
	def __init__(self, args):
		super(Critic, self).__init__()
		self.l2 = args.l2

		self.get_model(args)

	def get_model(self, args):
		inputs = tf.keras.Input(shape=(3, )) # My position (0, 1), what I payed so far (0, 1) and how much time I spent in queue (0, 1)

		z = tf.keras.layers.Dense(args.hidden_layer_critic, activation='relu')(inputs)
		z = tf.keras.layers.Dense(args.hidden_layer_critic, activation='relu')(z)
		value = tf.keras.layers.Dense(1, activation='linear')(z)[:, 0]

		self.model = tf.keras.Model(inputs=inputs, outputs=value)
		self.model.compile(loss=tf.keras.losses.MeanSquaredError(),
						   optimizer=tf.keras.optimizers.Adam(args.critic_learning_rate, global_clipnorm=args.clip_norm))

	@tf.function
	def train(self, x, targets):
		with tf.GradientTape() as tape:
			prediction = self.model(x)
			value_loss = self.model.compiled_loss(y_true=targets, y_pred=prediction)

			l2_loss = 0
			for var in self.model.trainable_variables:
				l2_loss += self.l2 * tf.nn.l2_loss(var)

			loss = value_loss + l2_loss

		self.model.optimizer.minimize(loss, self.model.trainable_variables, tape=tape)

	@tf.function(experimental_relax_shapes=True)
	def predict(self, x):
		return self.model(x)








