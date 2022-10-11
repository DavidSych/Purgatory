import os
import pickle
import numpy as np
from Simulations.Environment.agent import Agent
import copy, random


class Queue():
	def __init__(self, args):
		self.k = args.k
		self.F = args.F
		self.Q = args.Q
		self.T = args.T
		self.x_mean, self.x_std = args.x_mean, args.x_std
		self.args = args
		self.N_equal = (args.x_mean - args.k) * args.T

		self.step_num = 0

		self.leaving_time = np.zeros(shape=(args.T, ), dtype=np.int32)
		self.leaving_payment = np.zeros(shape=(args.Q + 1, ), dtype=np.int32)

	def initialize(self):
		self.agents = [Agent(self.args) for _ in range(self.x_mean)]
		# Payment (0), time (1), position (2), if was acting (3) and reward (4)
		self.state = np.zeros(shape=(self.x_mean, 5), dtype=np.int32)
		return self.state

	def update_state(self):
		for state, agent in zip(self.state, self.agents):
			agent.my_states[agent.t, :] = state

	def add_agents(self):
		to_add = int(np.random.normal(loc=self.x_mean, scale=self.x_std))
		for _ in range(to_add):
			self.agents.append(Agent(self.args))

		zeros = np.zeros(shape=(to_add, 5), dtype=np.int32)
		zeros[:, 2] = np.arange(to_add) + self.state.shape[0]
		print(self.state.shape, zeros.shape)
		self.state = np.concatenate([self.state, zeros], axis=0)

	def remove_agents(self):
		removed = []
		survived = np.where(self.state[:, 1] >= self.T)[0]
		paid = np.where(self.state[:, 0] >= self.F)[0]
		fined = np.arange(self.k)

		for i in survived:
			removed.append(self.agents[i])
			self.agents.pop(i)
		self.state = self.state[np.where(self.state[:, 1] < self.T)]

		for i in paid:
			removed.append(self.agents[i])
			self.agents.pop(i)
		self.state = self.state[np.where(self.state[:, 0] < self.F)]

		for i in fined:
			a = self.agents[i]
			removed.append(a)
			a.my_buffer[a.t-1, 4] -= self.Q
			self.agents.pop(i)
		self.state = self.state[self.k:]

		for r in removed:
			r.terminate()
			self.leaving_payment[r.payment] += 1
			self.leaving_time[r.t-1] += 1

		return removed

	@property
	def num_agents(self):
		return len(self.agents)

	def save(self, num):
		np.save(f'leaving_time_{num}.npy', self.leaving_time)
		np.save(f'leaving_payment_{num}.npy', self.leaving_payment)
		pickle.dump(self.args, open('args.pickle', 'wb'))

	def step(self, actions):
		'''
		Main method of the game, accepting actions for each agent and simulating a game day.

		:param actions: np.array[len_queue before step] of {0, ... F} of desired actions if agents doesn't forget
			   policy np.array[len queue before step, F+1] probability of actions for every stat. If not provided,
			   		the cost is not computed.
		:return: next_states: np.array[len_queue, 3] states in order [f, t, position]
				 rewards: np.array[len_queue before step] negative of action plus Q if fined in this step
				 costs: np.array[len_queue before step, F+1] negative of action + probability of fined * Q
		'''

		self.step_num += 1
		actions = actions.reshape((-1,)).astype(int)

		# Take actions unless you forget
		forgot_per_agent = np.random.uniform(0, 1, size=(len(self.agents, ))) <= np.array([a.p for a in self.agents])
		masked_actions = actions * forgot_per_agent

		self.state[:, 0] += masked_actions  # Increase payment of agents
		self.state[:, 1] += 1  # Increase time for each agent
		# Position will be modified later
		self.state[:, 3] = 1 - forgot_per_agent  # If he was acting
		self.state[:, 4] -= masked_actions  # Increase reward

		# Sort by current average payment
		average_payment = self.state[:, 0] / self.state[:, 1]
		new_order = np.argsort(average_payment)
		self.agents = [self.agents[i] for i in new_order]
		self.state = self.state[new_order]

		# Remove agents who paid enough, survived for long enough and are at the first k spots
		removed_agents = self.remove_agents()

		# Add new agents in the queue
		self.add_agents()

		return self.state[:, :3], removed_agents


