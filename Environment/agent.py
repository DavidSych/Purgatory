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

		# Storing agent's trajectory, (0) running payment, (1) my t, (2) my position,
		# (3) if I am acting and (4) my reward
		self.my_buffer = np.zeros((args.T, 5), dtype=np.int32)

	@property
	def average_payment(self):
		return self.payment / self.t

	def terminate(self):
		self.my_states = self.my_buffer[:self.t, :3]
		self.acting = self.my_buffer[:self.t, 3]
		self.my_rewards = self.my_buffer[:self.t, 4]


