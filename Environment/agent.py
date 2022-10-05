import numpy as np

class Agent:
	def __init__(self, args):
		self.t, self.payment = 0, 0
		if args.ignorance_distribution == 'uniform':
			self.p = np.random.uniform(args.p_min, 1)
		elif args.ignorance_distribution == 'beta':
			self.p = np.random.beta(args.alpha, args.beta)
		else:
			raise NotImplementedError(f'Unknown ignorance distribution {args.ignorance_distribution}.')

		# Storing agent's trajectory, (0) running payment, (1) my t, (2) my position
		self.my_states = np.zeros((args.T, 3), dtype=np.int)
		# If agent was allowed to act or not
		self.acting = np.zeros((args.T, ), np.int)
		# Storing agent's rewards and costs
		self.my_rewards = np.zeros(args.T)
		self.my_costs = np.zeros((args.T, args.F + 1))

	@property
	def average_payment(self):
		return self.payment / self.t

	def terminate(self):
		self.my_states = self.my_states[:self.t]
		self.my_rewards = self.my_rewards[:self.t]
		self.my_costs = self.my_costs[:self.t]


