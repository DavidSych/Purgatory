import numpy as np
from Simulations.Environment.queue import Queue
import argparse, os, shutil, datetime, pickle
from Simulations.Utils.misc import queue_saver, policy_saver

parser = argparse.ArgumentParser()
# TF params
parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility.")
parser.add_argument("--learning_rate", default=5e-2, type=float, help="Learning rate.")

parser.add_argument("--gamma", default=1, type=float, help="Return discounting.")
parser.add_argument("--epsilon", default=0.05, type=float, help="Exploration rate.")
parser.add_argument("--train_steps", default=1_000, type=int, help="How many simulations to train from.")
parser.add_argument("--train_sims", default=256, type=int, help="How many times to save progress.")

# Queue parameters
parser.add_argument("--F", default=4, type=int, help="End fine to pay.")
parser.add_argument("--Q", default=40, type=int, help="End fine to pay.")
parser.add_argument("--T", default=4, type=int, help="Time to survive in queue.")
parser.add_argument("--k", default=5, type=int, help="How many people have to pay in each step.")
parser.add_argument("--x_mean", default=100, type=float, help="Mean number of agents to add each step.")
parser.add_argument("--x_std", default=5, type=float, help="Standard deviation of the number of agents to add each step.")
parser.add_argument("--ignorance_distribution", default='fixed', type=str, help="What distribuin to use to sample probability of ignorance. Supported: fixed, uniform, beta.")
parser.add_argument("--p", default=0.5, type=float, help="Fixed probability of ignorance")
parser.add_argument("--p_min", default=0.5, type=float, help="Parameter of uniform distribution of ignorance.")
parser.add_argument("--alpha", default=2, type=float, help="Parameter of Beta distribution of ignorance.")
parser.add_argument("--beta", default=4, type=float, help="Parameter of Beta distribution of ignorance.")

args = parser.parse_args([] if "__file__" not in globals() else None)

np.random.seed(args.seed)
path = os.getcwd()
os.chdir(f'../Results/Q')
dir_name = f'{str(datetime.datetime.now().date())}_{str(datetime.datetime.now().time())[:5].replace(":", "-")}'
os.mkdir(dir_name)
os.chdir(dir_name)
pickle.dump(args, open("args.pickle", "wb"))

parent = '/'.join(path.split('/')[:-1])
shutil.copy(path + '/tabular_q.py', parent + '/Results/Q/' + dir_name)


def run(q_table):
	queue = Queue(args)
	state = queue.initialize()
	for sim in range(args.train_steps):
		qs = q_table[state[:, 0], state[:, 1], state[:, 2]]
		greedy_actions = np.argmax(qs, axis=-1)
		random_actions = np.random.randint(0, args.F, size=queue.num_agents)
		actions = np.where(np.random.uniform(size=queue.num_agents) > args.epsilon, greedy_actions, random_actions)

		next_state, removed = queue.step(actions)

		q_table = train(q_table, removed)

		state = next_state

	return queue, q_table


def train(q_table, removed):
	for r in removed:
		for t in range(r.t-1):
			s = r.my_states[t]
			reward = r.my_rewards[t]
			next_s = r.my_states[t + 1]
			next_q = (1 - r.p) * np.max(q_table[next_s[0], next_s[1], next_s[2], :])
			next_q += r.p * q_table[next_s[0], next_s[1], next_s[2], 0]  # If I forget, I play nothing
			current_q = q_table[s[0], s[1], s[2]].copy()

			target = (1 - args.learning_rate) * current_q
			target += reward + args.gamma * args.learning_rate * next_q
			q_table[s[0], s[1], s[2]] = target

		s = r.my_states[r.t-1]
		reward = r.my_rewards[r.t-1]
		current_q = q_table[s[0], s[1], s[2]].copy()

		target = (1 - args.learning_rate) * current_q + reward * args.gamma
		q_table[s[0], s[1], s[2]] = target

	return q_table


N_equal = (args.x_mean - args.k) * args.T
q_table = np.zeros(shape=(args.F, args.T, N_equal, args.F+1))

for i in range(args.train_sims):
	queue, q_table = run(q_table)

	flat_q_table = q_table.reshape((-1, args.F+1))
	policy = np.zeros_like(flat_q_table)
	greedy = np.argmax(q_table.reshape((-1, args.F+1)), axis=-1)
	policy[np.arange(policy.shape[0]), greedy] = 1 - args.epsilon
	policy[:, :] += args.epsilon / (args.F + 1)

	print(np.mean(policy, axis=0))
	policy = policy.reshape((args.F, args.T, queue.N_equal, args.F+1))

	policy_saver(policy, i)
	queue_saver(queue, i)
	print(f'Saving progress ({i+1}/{args.train_sims}).')

np.save('q_values.npy', q_table)





















