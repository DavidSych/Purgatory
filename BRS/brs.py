import numpy as np


class BRSAgent:
	def __init__(self, args):
		self.utilities = []
		self.F = args.F
		self.utility_samples = args.buffer_len

	@property
	def train_ready(self):
		return len(self.utilities) > self.utility_samples

	def actions(self, agents):
		for agent in agents:
			if agent.in_danger and agent.strategy < self.F:
				agent.strategy += 1
			elif agent.strategy > 0:
				agent.strategy -= 1
		actions = np.array([agent.strategy for agent in agents])
		return actions

	def train(self):
		pass

	def store(self, removed):
		self.utilities.append(np.sum(removed.my_rewards))

	def save(self, name, iteration):
		pass






















