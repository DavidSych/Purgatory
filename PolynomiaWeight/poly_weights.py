import numpy as np
from Simulations.Environment.queue import Queue
import argparse
import random

parser = argparse.ArgumentParser()
# TF params
parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility.")
parser.add_argument("--learning_rate", default=5e-2, type=float, help="Learning rate.")

parser.add_argument("--gamma", default=1, type=float, help="Return discounting.")
parser.add_argument("--epsilon", default=0.05, type=float, help="Exploration rate.")
parser.add_argument("--train_steps", default=4096, type=int, help="How many simulations to train from.")

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


def update(w_table, removed):
	for r in removed:
		for t in range(r.t):
			s = r.my_states[t]
			costs = r.my_costs[t]

			w_table[s[0], s[1], s[2], :] *= (1 - args.learning_rate * costs)


N_equal = (args.x_mean - args.k) * args.T
w_table = np.ones(shape=(args.F, args.T, N_equal, args.F + 1))

queue = Queue(args)
state = queue.initialize()
for _ in range(args.train_steps):
	ws = w_table[state[:, 0], state[:, 1], state[:, 2], :]
	ps = ws / np.sum(ws, axis=1, keepdims=True)

	actions = np.apply_along_axis(lambda p: random.choices(np.arange(args.F+1), weights=p), arr=ps, axis=1)

	next_state, removed = queue.step(actions)

	update(w_table, removed)

	state = next_state

queue.save(0)























