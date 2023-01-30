from Simulations.Environment.queue import Queue
from Simulations.Utils.misc import my_chdir
import numpy as np
import os, pickle, random
import matplotlib.pyplot as plt
import matplotlib as mpl


def sample_expectation(queue, steps, policy):
	visit_count = np.zeros(shape=(queue.F, queue.T, queue.N_equal, queue.F+1), dtype=np.int32)
	return_sum = np.zeros(shape=(queue.F, queue.T, queue.N_equal, queue.F+1), dtype=np.float32)

	state = queue.initialize()
	ps = policy[state[:, 0], state[:, 1], state[:, 2], :]
	actions = np.apply_along_axis(lambda p: random.choices(np.arange(args.F + 1), weights=p), arr=ps, axis=1)
	for _ in range(steps):
		state, removed = queue.step(actions)
		ps = policy[state[:, 0], state[:, 1], state[:, 2], :]
		actions = np.apply_along_axis(lambda p: random.choices(np.arange(args.F + 1), weights=p), arr=ps, axis=1)

		for r in removed:
			returns = np.sum(r.my_rewards)
			s = r.my_states[:, :3]
			a = np.mod(-r.my_rewards, queue.Q).astype(np.int64)
			return_sum[s[:, 0], s[:, 1], s[:, 2], a] += returns
			visit_count[s[:, 0], s[:, 1], s[:, 2], a] += 1

	return return_sum, visit_count


def mean_return(return_sum, visit_count):
	non_zero = np.where(visit_count > 0)
	return_sum[non_zero] = return_sum[non_zero] / visit_count[non_zero]
	return return_sum


def load_old(sim, return_sum, visit_count):
	root = os.getcwd()
	try:
		my_chdir('cce_data')
		old_returns = np.load(f'returns_{sim}.npy')
		old_counts = np.load(f'counts_{sim}.npy')
		return_sum += old_returns
		visit_count += old_counts
		print(f'Fount {np.sum(old_counts)} old data points.')
	except:
		print('No old data found.')

	np.save(f'returns_{sim}', return_sum)
	np.save(f'counts_{sim}', visit_count)

	os.chdir(root)
	return return_sum, visit_count


algorithm = 'PPO'
folder = '2022-10-20_22-13'
steps = 1000  # Steps in the queue to approximate everything with
sim_skip = 8  # Evaluate only every _ simulation


os.chdir(f'../Results/{algorithm}/{folder}')
args = pickle.load(open('args.pickle', 'rb'))
np.random.seed(args.seed)
if not os.path.isdir('figures'):
	os.mkdir('figures')

cutoffs = np.arange(100)
cut_regrets = np.zeros(shape=(args.train_sims // sim_skip, cutoffs.shape[0]))

cmap = mpl.cm.get_cmap('Spectral')
plt.grid(0.25)

for sim in range(len(os.listdir(os.getcwd() + '/policies')) // sim_skip):
	print(f'Working on simulation {sim * sim_skip + 1}.')
	queue = Queue(args)
	policy = np.load(os.getcwd() + f'/policies/policy_{sim * sim_skip}.npy')

	return_sum, counts = sample_expectation(queue, steps, policy)
	return_sum, counts = load_old(sim * sim_skip, return_sum, counts)
	mean_returns = mean_return(return_sum, counts)

	mean_returns = mean_returns.reshape((-1, args.F + 1))
	counts = counts.reshape((-1, args.F + 1))

	u, c = np.unique(counts, return_counts=True)
	u = u[np.argsort(c)[::-1]]
	c = np.sort(c)[::-1]

	flat_policy = policy.reshape((-1, args.F + 1))
	expected_utility = np.sum(mean_returns * flat_policy, axis=1, keepdims=True)
	regrets = mean_returns - expected_utility

	for i, c in enumerate(cutoffs):
		low_visit_count = np.where(counts < c)
		regrets[low_visit_count] = 0
		cut_regrets[sim, i] = np.max(regrets)

	plt.plot(cutoffs, cut_regrets[sim], c=cmap(sim * sim_skip / args.train_sims))

plt.ylabel('Max Regret ($\epsilon$)')
plt.xlabel('Minimum count')
if cut_regrets[0, -1] > 0:
	plt.ylim((0, 1.5 * cut_regrets[0, -1]))
cbar = plt.colorbar(mpl.cm.ScalarMappable(cmap=cmap), ax=plt.gca())
cbar.set_ticks(np.arange(8) / 7)
cbar.set_ticklabels(np.arange(0, args.train_sims, (args.train_sims - 1) / 7).astype(np.int32) + 1)
plt.savefig(os.getcwd() + '/figures/regrets.pdf')

plt.yscale('log')
plt.savefig(os.getcwd() + '/figures/regrets_log.pdf')
plt.clf()

print(f'Final epsilon is {cut_regrets[-1, -1]}')
print(f'Minimum encountered epsilon is {np.min(cut_regrets)}')










