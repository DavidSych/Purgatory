from Simulations.Environment.queue import Queue
import numpy as np
import torch
import os, pickle, random
import matplotlib.pyplot as plt
import matplotlib as mpl


def sample_expectation(queue, samples, policy):
	visit_count = np.zeros(shape=(queue.F, queue.T, queue.N_equal, queue.F+1), dtype=np.int32)
	return_sum = np.zeros(shape=(queue.F, queue.T, queue.N_equal, queue.F+1), dtype=np.float32)

	state = queue.initialize()
	ps = policy[state[:, 0], state[:, 1], state[:, 2], :]
	actions = np.apply_along_axis(lambda p: random.choices(np.arange(args.F + 1), weights=p), arr=ps, axis=1)
	print(actions.shape)
	for _ in range(steps):
		state, removed = queue.step(actions)
		actions = policy[state[:, 0], state[:, 1], state[:, 2]]

		for r in removed:
			returns = np.sum(r.my_rewards)
			s = r.my_states
			a = np.mod(-r.my_rewards, queue.Q)
			return_sum[s[:, 0], s[:, 1], s[:, 2], a] += returns
			visit_count[s[:, 0], s[:, 1], s[:, 2], a] += 1

	return return_sum, visit_count


def mean_return(return_sum, visit_count):
	non_zero = np.where(visit_count > 0)
	return_sum[non_zero] = return_sum[non_zero] / visit_count[non_zero]
	return return_sum


def load_old(sim, return_sum, visit_count):
	try:
		old_returns = np.load(f'returns_{sim}.npy')
		old_counts = np.load(f'counts_{sim}.npy')
		return_sum += old_returns
		visit_count += old_counts
		print(f'Fount {np.sum(old_counts)} old data points.')
	except:
		print('No old data found.')

	np.save(f'returns_{sim}', return_sum)
	np.save(f'counts_{sim}', visit_count)
	return return_sum, visit_count


algorithm = 'PW'
dir_name = '2022-10-04_11-30'
steps = 1_000  # Steps in the queue to approximate everything with

os.chdir(f'../Results/{algorithm}/{dir_name}')
args = pickle.load(open('args.pickle', 'rb'))

cutoffs = np.arange(500)
cut_regrets = np.zeros(shape=(args.train_sims, cutoffs.shape[0]))

cmap = mpl.cm.get_cmap('Spectral')

for sim in range(args.train_sims):
	print(f'Working on simulation {sim + 1}.')
	queue = Queue(args, dir_name)
	policy = np.load(f'policy_{sim}.npy')

	return_sum, counts = sample_expectation(queue, steps, policy)
	return_sum, counts = load_old(sim, return_sum, counts)
	mean_returns = mean_return(return_sum, counts)

	mean_returns = mean_returns.reshape((args.g, -1, args.F + 1))
	counts = counts.reshape((args.g, -1, args.F + 1))

	u, c = np.unique(counts, return_counts=True)
	u = u[np.argsort(c)[::-1]]
	c = np.sort(c)[::-1]

	expected_utility = np.sum(mean_returns * policy, axis=2, keepdims=True)
	regrets = mean_returns - expected_utility

	for i, c in enumerate(cutoffs):
		low_visit_count = np.where(counts < c)
		regrets[low_visit_count] = 0
		cut_regrets[sim, i] = np.max(regrets)

	plt.plot(cutoffs, cut_regrets[sim], c=cmap(sim / args.train_sims))

plt.ylabel('Max Regret ($\epsilon$)')
plt.xlabel('Minimum count')
plt.ylim((0, 1.5 * cut_regrets[0, -1]))
cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap))
cbar.set_ticks(np.arange(8) / 7)
cbar.set_ticklabels(np.arange(0, args.train_sims, (args.train_sims - 1) / 7).astype(np.int32) + 1)
plt.savefig('regrets.png')
plt.clf()

print(f'Final epsilon is {cut_regrets[-1, -1]}')
print(f'Minimum encountered epsilon is {np.min(cut_regrets)}')










