import os

import numpy as np
from Environment.queue import Queue
import argparse
import random, datetime
import pickle, shutil, time
from Utils.misc import queue_saver, policy_saver

parser = argparse.ArgumentParser()
# TF params
parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility.")
parser.add_argument("--learning_rate", default=1e-1, type=float, help="Learning rate.")
parser.add_argument("--min_visit_count", default=1, type=int, help="Minimum number of visits to consider estimated cost during policy update")

parser.add_argument("--train_steps", default=1_000, type=int, help="How many simulations to train for.")
parser.add_argument("--train_sims", default=65, type=int, help="How many times to save progress.")

# Queue parameters
parser.add_argument("--F", default=4, type=int, help="End fine to pay.")
parser.add_argument("--Q", default=4, type=int, help="End fine to pay.")
parser.add_argument("--T", default=1, type=int, help="Time to survive in queue.")
parser.add_argument("--k", default=0, type=int, help="How many people have to pay in each step.")
parser.add_argument("--x_mean", default=100, type=float, help="Mean number of agents to add each step.")
parser.add_argument("--x_std", default=5, type=float, help="Standard deviation of the number of agents to add each step.")
parser.add_argument("--ignorance_distribution", default='uniform', type=str, help="What distribuin to use to sample probability of ignorance.")
parser.add_argument("--p_min", default=0.5, type=float, help="Parameter of uniform distribution of ignorance.")
parser.add_argument("--alpha", default=2, type=float, help="Parameter of Beta distribution of ignorance.")
parser.add_argument("--beta", default=4, type=float, help="Parameter of Beta distribution of ignorance.")

args = parser.parse_args([] if "__file__" not in globals() else None)

np.random.seed(args.seed)
path = os.getcwd()
os.chdir(f'../Results/PW')
dir_name = f'{str(datetime.datetime.now().date())}_{str(datetime.datetime.now().time())[:5].replace(":", "-")}'
os.mkdir(dir_name)
os.chdir(dir_name)
pickle.dump(args, open("args.pickle", "wb"))

parent = '/'.join(path.split('/')[:-1])
shutil.copy(path + '/poly_weights.py', parent + '/Results/PW/' + dir_name)


def run(args, w_table):
	tot_costs = np.zeros(w_table.shape, dtype=np.float32)
	tot_counts = np.zeros(w_table.shape, dtype=np.int32)
	queue = Queue(args)
	state = queue.initialize()
	sims = args.train_steps
	for sim in range(sims):
		ws = w_table[state[:, 0], state[:, 1], state[:, 2], :]
		ps = ws / np.sum(ws, axis=1, keepdims=True)

		actions = np.apply_along_axis(lambda p: random.choices(np.arange(args.F+1), weights=p), arr=ps, axis=1)

		state, removed = queue.step(actions)

		for r in removed:
			states = r.my_states
			actions = np.mod(r.my_rewards, args.Q).astype(int)
			cost = - np.cumsum(r.my_rewards[::-1])[::-1]
			cost /= args.Q + args.F - 1  # Normalize cost to [0, 1]
			for s, a, c in zip(states, actions, cost):
				tot_costs[s[0], s[1], s[2], a] += c
				tot_counts[s[0], s[1], s[2], a] += 1

	visited = np.where(tot_counts >= args.min_visit_count)
	tot_costs[visited] = tot_costs[visited] / tot_counts[visited]
	return tot_costs, queue


def update(w_table, costs):
	w_table *= (1 - args.learning_rate * costs)
	return w_table


N_equal = (args.x_mean - args.k) * args.T
w_table = np.ones(shape=(args.F, args.T, N_equal, args.F + 1))

for i in range(args.train_sims):
	costs, queue = run(args, w_table)
	w_table = update(w_table, costs)

	policy = w_table / np.sum(w_table, axis=-1, keepdims=True)

	policy_saver(policy, i)
	queue_saver(queue, i)
	print(f'Saving progress ({i+1}/{args.train_sims}).')

np.save('weights.npy', w_table)





















