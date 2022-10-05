import os
import pickle
import numpy as np
from Simulations.Environment.agent import Agent
import copy, random


class Queue():
	def __init__(self, args, full_cost=False):
		self.k = args.k
		self.F = args.F
		self.Q = args.Q
		self.T = args.T
		self.full_cost = full_cost  # If we want to compute cost of every action
		self.x_mean, self.x_std = args.x_mean, args.x_std
		self.args = args
		self.N_equal = (args.x_mean - args.k) * args.T

		self.step_num = 0

		self.leaving_time = np.zeros(shape=(args.T, ), dtype=np.int)
		self.leaving_payment = np.zeros(shape=(args.Q + 1, ), dtype=np.int)

		if full_cost:
			self.cost_samples = args.cost_samples

	def initialize(self):
		self.agents = [Agent(self.args) for _ in range(self.x_mean)]
		return self.state()

	def state(self):
		state = []
		for position, agent in enumerate(self.agents):
			s = [agent.payment, agent.t, min(position, self.N_equal - 1)]
			state.append(s)
			agent.my_states[agent.t, :] = s

		return np.array(state, dtype=np.int)

	def add_agents(self):
		to_add = int(np.random.normal(loc=self.x_mean, scale=self.x_std))
		for _ in range(to_add):
			new_agent = Agent(self.args)
			self.agents.append(new_agent)

	def remove_agents(self):
		removed = []
		for i in reversed(range(len(self.agents))):
			if self.agents[i].t >= self.T or self.agents[i].payment >= self.F:
				removed.append(self.agents[i])
				self.agents.pop(i)

		for i in reversed(range(min(self.k, len(self.agents)))):
			a = self.agents[i]
			removed.append(a)
			a.payment += self.Q
			a.my_rewards[a.t] -= self.Q
			self.agents.pop(i)

		for r in removed:
			r.terminate()
			self.leaving_payment[r.payment] += 1
			self.leaving_time[r.t-1] += 1

		return removed

	def approximate_costs(self, policy):
		costs = np.zeros(shape=(self.cost_samples, len(self.agents), self.F+1))
		for i, agent in enumerate(self.agents):
			for a in range(self.F+1):
				for t in range(self.cost_samples):
					queue_copy = copy.deepcopy(self)
					tracked_agent = queue_copy.agents[i]
					starting_reward = np.sum(tracked_agent.my_rewards)

					state, removed = queue_copy.state(), []
					ps = policy[state[:, 0], state[:, 1], state[:, 2], :]
					actions = np.apply_along_axis(lambda p: random.choices(np.arange(self.F + 1), weights=p), arr=ps, axis=1)
					actions[i] = a  # We fix the first action and let the agent sample the rest

					state, removed = queue_copy.step(actions)

					while not tracked_agent in removed:
						ps = policy[state[:, 0], state[:, 1], state[:, 2], :]
						actions = np.apply_along_axis(lambda p: random.choices(np.arange(self.F+1), weights=p), arr=ps, axis=1)
						state, removed = queue_copy.step(actions)

					cost = - (np.sum(tracked_agent.my_rewards) - starting_reward)
					costs[t, i, a] = cost

		return np.mean(costs, axis=0)

	@property
	def num_agents(self):
		return len(self.agents)

	def save(self, num):
		root = os.getcwd()
		os.chdir('../Results')
		np.save(f'leaving_time_{num}.npy', self.leaving_time)
		np.save(f'leaving_payment_{num}.npy', self.leaving_payment)
		pickle.dump(self.args, open('args.pickle', 'wb'))
		os.chdir(root)

	def step(self, actions, policy=None):
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

		# Compute rewards and approximate costs
		if policy is not None:
			costs = self.approximate_costs(policy)
			for c, agent in zip(costs, self.agents):
				agent.my_costs[agent.t, :] = c

		# Take actions unless you forget
		forgot_per_agent = np.random.uniform(0, 1, size=(len(self.agents, ))) <= np.array([a.p for a in self.agents])
		for forgot, agent, action in zip(forgot_per_agent, self.agents, actions.reshape((-1,))):
			if not forgot:
				action = min(action, self.F - agent.payment)  # I will not overpay
				agent.payment += action
				agent.my_rewards[agent.t] = - action
				agent.acting[agent.t] = 1
			agent.t += 1

		# Sort by current average payment
		new_order = np.argsort([agent.average_payment for agent in self.agents])
		self.agents = [self.agents[i] for i in new_order]

		# Remove agents who paid enough, survived for long enough and are at the first k spots
		removed_agents = self.remove_agents()

		# Add new agents in the queue
		self.add_agents()

		# Return current queue state for each agent and rewards & costs in the order of received `actions`
		current_state = self.state()

		return current_state, removed_agents


