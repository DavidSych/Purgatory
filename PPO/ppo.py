import numpy as np
from PPO.network import Actor, Critic
import random, os
import torch
from Utils.misc import policy_saver, my_chdir


class PPOLearner:
	def __init__(self, args):
		self.actor = Actor(args)
		self.critic = Critic(args)

		self.k = args.k
		self.F = args.F
		self.Q = args.Q
		self.T = args.T
		self.x_mean = args.x_mean
		self.N_equal = (args.x_mean - args.k) * args.T

		# (0-2) state, (3) action, (4) returns, (5) advantage, (6) probability of ignorance
		self.buffer = np.zeros(shape=(args.buffer_len, 7), dtype=np.float32)
		self.pointer = 0
		self.train_cycles = args.train_cycles
		self.gamma, self._lambda = args.gamma, args._lambda

	@property
	def train_ready(self):
		return self.pointer >= self.buffer.shape[0]

	def preprocess(self, state):
		state = state.astype(np.float32)
		state[:, 0] = state[:, 0] / self.F
		state[:, 1] = state[:, 1] / self.T
		state[:, 2] = state[:, 2] / self.N_equal
		return state

	def actions(self, observation):
		inputs = self.preprocess(observation)
		policy = self.actor(torch.tensor(inputs)).detach().numpy()
		actions = np.apply_along_axis(lambda p: random.choices(np.arange(self.F + 1), weights=p), arr=policy, axis=1)
		return actions

	def train(self):
		states = torch.tensor(self.buffer[:, :3].astype(np.float32))
		actions = torch.tensor(self.buffer[:, 3].astype(np.int64))
		p = torch.tensor(self.buffer[:, 6].astype(np.float32))
		policy = self.actor(states).detach() * (1 - p)[:, None]
		policy[:, 0] += p
		old_probs = policy[torch.arange(states.shape[0]), actions]
		returns = torch.tensor(self.buffer[:, 4].astype(np.float32))
		advantage = torch.tensor(self.buffer[:, 5].astype(np.float32))

		for _ in range(self.train_cycles):
			self.critic.train_iteration(states, returns)
			self.actor.train_iteration(states, advantage, actions, old_probs, p)

		self.pointer = 0

	def store(self, removed):
		t_steps = np.arange(removed.t)
		r_actions = removed.my_actions
		r_states = self.preprocess(removed.my_states)
		rewards = removed.my_rewards / (self.F + self.Q - 1)  # Scale the rewards to [-1,0]
		returns = rewards * (self.gamma ** t_steps)
		returns = np.cumsum(returns[::-1])[::-1] / (self.gamma ** t_steps)

		values = self.critic(torch.tensor(r_states)).detach().numpy()
		values = np.append(values, 0)
		td_error = rewards + self.gamma * values[1:] - values[:-1]
		decay_factor = self.gamma * self._lambda
		adv = td_error * (decay_factor ** t_steps)
		adv = np.cumsum(adv[::-1])[::-1] / (decay_factor ** t_steps)

		to_add = min(removed.t, self.buffer.shape[0] - self.pointer)

		self.buffer[self.pointer:self.pointer+to_add, :3] = r_states[:to_add]
		self.buffer[self.pointer:self.pointer+to_add, 3] = r_actions[:to_add]
		self.buffer[self.pointer:self.pointer+to_add, 4] = returns[:to_add]
		self.buffer[self.pointer:self.pointer+to_add, 5] = adv[:to_add]
		self.buffer[self.pointer:self.pointer+to_add, 6] = removed.p * np.ones(to_add)

		self.pointer += to_add

	def save(self, name, iteration):
		root = os.getcwd()
		my_chdir(name)

		all_states = self.preprocess(np.mgrid[0:self.F:1, 0:self.T:1, 0:self.N_equal:1]).transpose((1, 2, 3, 0)).reshape(-1, 3)
		policy = self.actor(torch.tensor(all_states.astype(np.float32))).detach().numpy()
		policy_saver(policy.reshape((self.F, self.T, self.N_equal, self.F + 1)), iteration)

		os.chdir(root)






















