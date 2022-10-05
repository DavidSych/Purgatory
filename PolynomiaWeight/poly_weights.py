import os

import numpy as np
from Simulations.Environment.queue import Queue
import argparse
import random, datetime
import pickle, shutil, time

parser = argparse.ArgumentParser()
# TF params
parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility.")
parser.add_argument("--learning_rate", default=1e-1, type=float, help="Learning rate.")
parser.add_argument("--cost_samples", default=10, type=int, help="Monte carlo simulations to approximate cost of an action.")

parser.add_argument("--train_steps", default=1_000, type=int, help="How many simulations to train for.")
parser.add_argument("--train_sims", default=1, type=int, help="How many times to save progress.")

# Queue parameters
parser.add_argument("--F", default=4, type=int, help="End fine to pay.")
parser.add_argument("--Q", default=4, type=int, help="End fine to pay.")
parser.add_argument("--T", default=4, type=int, help="Time to survive in queue.")
parser.add_argument("--k", default=5, type=int, help="How many people have to pay in each step.")
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


def run(w_table):
	queue = Queue(args, full_cost=True)
	state = queue.initialize()
	sims = args.train_steps
	for sim in range(sims):
		ws = w_table[state[:, 0], state[:, 1], state[:, 2], :]
		ps = ws / np.sum(ws, axis=1, keepdims=True)

		actions = np.apply_along_axis(lambda p: random.choices(np.arange(args.F+1), weights=p), arr=ps, axis=1)

		next_state, removed = queue.step(actions, w_table / np.sum(w_table, axis=3, keepdims=True))

		update(w_table, removed)

		state = next_state

	return queue


def update(w_table, removed):
	for r in removed:
		for t in range(r.t):
			s = r.my_states[t]
			costs = r.my_costs[t] / (args.F + args.Q)  # Scale to [0, 1]

			w_table[s[0], s[1], s[2], :] *= (1 - args.learning_rate * costs)


N_equal = (args.x_mean - args.k) * args.T
w_table = np.ones(shape=(args.F, args.T, N_equal, args.F + 1))

for i in range(args.train_sims):
	run(w_table)

	policy = w_table / np.sum(w_table, axis=-1, keepdims=True)
	np.save(f'policy_{i}.npy', policy)
	print(f'Saving progress ({i+1}/{args.train_sims}).')

np.save('weights.npy', w_table)





















