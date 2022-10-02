import numpy as np
from Simulations.Environment.queue import Queue
import argparse

parser = argparse.ArgumentParser()
# TF params
parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility.")
parser.add_argument("--learning_rate", default=5e-2, type=float, help="Learning rate.")

parser.add_argument("--gamma", default=1, type=float, help="Return discounting.")
parser.add_argument("--epsilon", default=0.05, type=float, help="Exploration rate.")
parser.add_argument("--train_steps", default=128, type=int, help="How many simulations to train from.")

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


def run(evaluate):
	queue = Queue(args)
	state = queue.initialize()
	for sim in range(args.train_steps):
		qs = q_table[state[:, 0], state[:, 1], state[:, 2]]
		greedy_actions = np.argmax(qs, axis=-1)
		random_actions = np.random.randint(0, args.F, size=queue.num_agents)
		actions = np.where(np.random.uniform(size=queue.num_agents) > args.epsilon, greedy_actions, random_actions)

		next_state, removed = queue.step(actions)

		if not evaluate:
			train(q_table, removed)

		state = next_state

	return queue


def train(q_table, removed):
	for r in removed:
		for t in range(r.t-1):
			s = r.my_states[t]
			reward = r.my_rewards[t]
			next_s = r.my_states[t + 1]
			next_q = np.max(q_table[next_s[0], next_s[1], next_s[2], :])
			current_q = q_table[s[0], s[1], s[2]].copy()

			target = (1 - args.learning_rate) * current_q
			target += reward + args.gamma * args.learning_rate * next_q
			q_table[s[0], s[1], s[2]] = target

		s = r.my_states[r.t-1]
		reward = r.my_rewards[r.t-1]
		current_q = q_table[s[0], s[1], s[2]].copy()

		target = (1 - args.learning_rate) * current_q + reward
		q_table[s[0], s[1], s[2]] = target


N_equal = (args.x_mean - args.k) * args.T
q_table = np.zeros(shape=(args.F, args.T, N_equal, args.F + 1))

run(evaluate=False)
queue = run(evaluate=True)

queue.save(0)
np.save('q_values.npy', q_table)























