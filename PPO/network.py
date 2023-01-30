import numpy as np
import torch


class Actor(torch.nn.Module):
	def __init__(self, args):
		super(Actor, self).__init__()
		self.eps = args.epsilon
		self.l2 = args.l2
		self.c_entropy = args.entropy_weight

		ignorance_dist = torch.zeros(args.F + 1)
		ignorance_dist[0] = 1
		self.ign_dist = ignorance_dist

		self.linear_1 = torch.nn.Linear(3, args.hidden_layer_actor)
		self.relu = torch.nn.ReLU()
		self.linear_2 = torch.nn.Linear(args.hidden_layer_actor, args.F+1)
		self.softmax = torch.nn.Softmax(dim=-1)

		self.optimizer = torch.optim.Adam(self.parameters(), lr=args.actor_learning_rate)

	def forward(self, x):
		x = self.linear_1(x)
		x = self.relu(x)
		x = self.linear_2(x)
		return self.softmax(x)

	def train_iteration(self, x, advantage, actions, old_prob, p):
		policy = (1 - p)[:, None] * self.forward(x) + p[:, None] * self.ign_dist
		probs = policy[torch.arange(policy.shape[0]), actions]

		ratio = probs / old_prob

		clipped_advantage = torch.where(advantage > 0, (1 + self.eps) * advantage, (1 - self.eps) * advantage)

		loss = - torch.mean(torch.minimum(ratio * advantage, clipped_advantage))

		loss += self.c_entropy * torch.mean(policy * torch.log(policy))

		for var in self.parameters():
			loss += self.l2 * torch.norm(var)

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()


class Critic(torch.nn.Module):
	def __init__(self, args):
		super(Critic, self).__init__()
		self.l2 = args.l2

		self.linear_1 = torch.nn.Linear(3, args.hidden_layer_critic)
		self.relu1 = torch.nn.ReLU()
		self.linear_2 = torch.nn.Linear(args.hidden_layer_critic, args.hidden_layer_critic)
		self.relu2 = torch.nn.ReLU()
		self.linear_3 = torch.nn.Linear(args.hidden_layer_critic, 1)

		self.optimizer = torch.optim.Adam(self.parameters(), lr=args.critic_learning_rate)

	def forward(self, x):
		y = self.linear_1(x)
		y = self.relu1(y)
		y = self.linear_2(y)
		y = self.relu2(y)
		y = self.linear_3(y)
		return y[:, 0]

	def train_iteration(self, x, y):
		value = self.forward(x)
		loss = torch.nn.MSELoss()(value, y)
		for var in self.parameters():
			loss += self.l2 * torch.norm(var)

		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
