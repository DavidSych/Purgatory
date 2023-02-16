import numpy as np
import random


class TabularAgent:
	def __init__(self, tabular_policy, utility_samples):
		self.policy = tabular_policy
		self.utilities = []
		self.utility_samples = utility_samples

	@property
	def train_ready(self):
		return len(self.utilities) > self.utility_samples

	def actions(self, observation):
		o0, o1, o2 = observation[:, 0], observation[:, 1], observation[:, 2]
		policy = self.policy[o0, o1, o2]
		actions = np.apply_along_axis(lambda p: random.choices(np.arange(self.policy.shape[-1]), weights=p), arr=policy, axis=1)
		return actions

	def train(self):
		pass

	def store(self, removed):
		self.utilities.append(np.sum(removed.my_rewards))

	def save(self, name, iteration):
		raise Exception('Trying to save a dummy tabular agent.')






















