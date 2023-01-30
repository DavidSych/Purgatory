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

		self.leaving_time = np.zeros(shape=(args.T, ), dtype=np.int)
		self.leaving_payment = np.zeros(shape=(args.Q + args.F, ), dtype=np.int)

	def initialize(self):
		self.agents = [Agent(self.args) for _ in range(self.x_mean)]
		return self.state()

	def state(self):
		state = np.empty(shape=(self.num_agents, 3), dtype=int)
		for position, agent in enumerate(self.agents):
			s = (agent.payment, agent.t-1, min(position, self.N_equal - 1))
			state[position, :] = s
			agent.my_states[agent.t-1, :] = s

		return state

	def add_agents(self):
		to_add = int(np.random.normal(loc=self.x_mean, scale=self.x_std))
		for _ in range(to_add):
			new_agent = Agent(self.args)
			self.agents.append(new_agent)

	def remove_agents(self):
		removed = []
		# Remove those who paid enough
		for i in reversed(range(len(self.agents))):
			if self.agents[i].payment >= self.F:
				removed.append(self.agents[i])
				self.agents.pop(i)

		# Fine first `k` agents
		for i in reversed(range(min(self.k, len(self.agents)))):
			a = self.agents[i]
			removed.append(a)
			a.payment += self.Q
			a.my_rewards[a.t-1] -= self.Q
			self.agents.pop(i)

		# Remove those who survived long enough
		for i in reversed(range(len(self.agents))):
			if self.agents[i].t >= self.T:
				removed.append(self.agents[i])
				self.agents.pop(i)

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
		for forgot, agent, action in zip(forgot_per_agent, self.agents, actions):
			if not forgot:
				action = min(action, self.F - agent.payment)  # I will not overpay
				agent.payment += action
				agent.my_rewards[agent.t] = - action
				agent.acting[agent.t] = 1
			agent.my_actions[agent.t] = action
			agent.t += 1

		# Sort by current average payment
		self.agents.sort(key=lambda agent: agent.average_payment, reverse=False)

		# Remove agents who paid enough, survived for long enough and are at the first k spots
		removed_agents = self.remove_agents()

		# Add new agents in the queue
		self.add_agents()

		# Return current queue state for each agent and rewards & costs in the order of received `actions`
		current_state = self.state()

		return current_state, removed_agents


