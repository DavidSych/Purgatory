import numpy as np
from Simulations.Environment.queue import Queue
from network import Actor, Critic
import argparse, datetime, shutil
import pickle, random, os
import torch
from Simulations.Utils.misc import queue_saver, policy_saver

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

parser = argparse.ArgumentParser()
# TF params
parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility.")
parser.add_argument("--threads", default=15, type=int, help="Number of CPU threads to use.")
parser.add_argument("--hidden_layer_actor", default=4, type=int, help="Size of the hidden layer of the network.")
parser.add_argument("--hidden_layer_critic", default=32, type=int, help="Size of the hidden layer of the network.")
parser.add_argument("--actor_learning_rate", default=3e-4, type=float, help="Learning rate.")
parser.add_argument("--critic_learning_rate", default=1e-3, type=float, help="Learning rate.")
parser.add_argument("--entropy_weight", default=0, type=float, help="Entropy regularization constant.")
parser.add_argument("--l2", default=0, type=float, help="L2 regularization constant.")
parser.add_argument("--clip_norm", default=0.1, type=float, help="Gradient clip norm.")

parser.add_argument("--buffer_len", default=10_000, type=int, help="Number of time steps to train on.")
parser.add_argument("--epsilon", default=0.05, type=float, help="Clipping constant.")
parser.add_argument("--gamma", default=1., type=float, help="Return discounting.")
parser.add_argument("--_lambda", default=0.97, type=float, help="Advantage discounting.")
parser.add_argument("--train_cycles", default=32, type=int, help="Number of PPO passes.")
parser.add_argument("--train_sims", default=1024, type=int, help="How many simulations to train from.")
parser.add_argument("--evaluate", default=False, type=bool, help="If NashConv should be computed as well.")

# Queue parameters
parser.add_argument("--F", default=4, type=int, help="Amount to pay to leave queue.")
parser.add_argument("--Q", default=6, type=int, help="End fine to pay.")
parser.add_argument("--T", default=4, type=int, help="Time to survive in queue.")
parser.add_argument("--k", default=2, type=int, help="How many people have to pay in each step.")
parser.add_argument("--g", default=1, type=int, help="How many groups to use.")
parser.add_argument("--tau", default=1, type=int, help="Don't use agents added to queue before tau * T steps.")
parser.add_argument("--x_mean", default=100, type=float, help="Mean number of agents to add each step.")
parser.add_argument("--x_std", default=0, type=float, help="Standard deviation of the number of agents to add each step.")
parser.add_argument("--N_init", default=20, type=int, help="Initial number of agents.")
parser.add_argument("--ignorance_distribution", default='fixed', type=str, help="What distribuin to use to sample probability of ignorance. Supported: fixed, uniform, beta.")
parser.add_argument("--p", default=0.5, type=float, help="Fixed probability of ignorance")
parser.add_argument("--p_min", default=0.0, type=float, help="Parameter of uniform distribution of ignorance.")
parser.add_argument("--alpha", default=2, type=float, help="Parameter of Beta distribution of ignorance.")
parser.add_argument("--beta", default=4, type=float, help="Parameter of Beta distribution of ignorance.")
parser.add_argument("--reward_shaping", default=False, type=bool, help="If rewards shaping should be used.")

args = parser.parse_args([] if "__file__" not in globals() else None)

np.random.seed(args.seed)
torch.manual_seed(0)
path = os.getcwd()
os.chdir(f'../Results/PPO')
dir_name = f'{str(datetime.datetime.now().date())}_{str(datetime.datetime.now().time())[:5].replace(":", "-")}'
os.mkdir(dir_name)
os.chdir(dir_name)
pickle.dump(args, open("args.pickle", "wb"))

parent = '/'.join(path.split('/')[:-1])
shutil.copy(path + '/ppo.py', parent + '/Results/PPO/' + dir_name)


def train(buffer, actor, critic):
	states = torch.tensor(buffer[:, :3].astype(np.float32))
	actions = torch.tensor(buffer[:, 3].astype(np.int64))
	p = torch.tensor(buffer[:, 6].astype(np.float32))
	policy = actor(states).detach() * (1 - p)[:, None]
	policy[:, 0] += p
	old_probs = policy[torch.arange(states.shape[0]), actions]
	returns = torch.tensor(buffer[:, 4].astype(np.float32))
	advantage = torch.tensor(buffer[:, 5].astype(np.float32))

	for _ in range(args.train_cycles):
		critic.train_iteration(states, returns)
		actor.train_iteration(states, advantage, actions, old_probs, p)


def preprocess(state):
	state = state.astype(np.float32)
	state[:, 0] = state[:, 0] / args.F
	state[:, 1] = state[:, 1] / args.T
	state[:, 2] = state[:, 2] / ((args.x_mean - args.k) * args.T)
	return state

actor = Actor(args)
critic = Critic(args)

for i in range(args.train_sims):
	queue = Queue(args)
	state = queue.initialize()
	# (0-2) state, (3) action, (4) returns, (5) advantage, (6) probability of ignorance
	buffer = np.zeros(shape=(args.buffer_len, 7), dtype=np.float32)
	pointer = 0  # Where to write in the buffer

	# Warm-up to the equilibrium length
	for _ in range(args.tau * args.T):
		state = preprocess(state)
		policy = actor(torch.tensor(state)).detach().numpy()
		actions = np.apply_along_axis(lambda p: random.choices(np.arange(args.F+1), weights=p), arr=policy, axis=1)
		state, removed = queue.step(actions)

	# Gather data and perform training
	while pointer < args.buffer_len:
		state = preprocess(state)
		policy = actor(torch.tensor(state)).detach().numpy()
		actions = np.apply_along_axis(lambda p: random.choices(np.arange(args.F+1), weights=p), arr=policy, axis=1)

		state, removed = queue.step(actions)

		for r in removed:
			t_steps = np.arange(r.t)
			#is_acting = np.where(r.acting > 0)
			r_actions = r.my_actions
			r_states = preprocess(r.my_states)
			rewards = r.my_rewards / (args.F + args.Q - 1)  # Scale the rewards to [-1,0]
			returns = rewards * (args.gamma ** t_steps)
			returns = np.cumsum(returns[::-1])[::-1] / (args.gamma ** t_steps)

			values = critic(torch.tensor(r_states)).detach().numpy()
			values = np.append(values, 0)
			td_error = rewards + args.gamma * values[1:] - values[:-1]
			decay_factor = args.gamma * args._lambda
			adv = td_error * (decay_factor ** t_steps)
			adv = np.cumsum(adv[::-1])[::-1] / (decay_factor ** t_steps)

			to_add = min(r.t, args.buffer_len - pointer)

			buffer[pointer:pointer+to_add, :3] = r_states[:to_add]
			buffer[pointer:pointer+to_add, 3] = r_actions[:to_add]
			buffer[pointer:pointer+to_add, 4] = returns[:to_add]
			buffer[pointer:pointer+to_add, 5] = adv[:to_add]
			buffer[pointer:pointer+to_add, 6] = r.p * np.ones(to_add)

			pointer += to_add

	all_states = preprocess(np.mgrid[0:queue.F:1, 0:queue.T:1, 0:queue.N_equal:1]).transpose((1, 2, 3, 0)).reshape(-1, 3)
	policy = actor(torch.tensor(all_states.astype(np.float32))).detach().numpy()

	train(buffer, actor, critic)

	policy_saver(policy.reshape((queue.F, queue.T, queue.N_equal, queue.F+1)), i)
	queue_saver(queue, i)
	print(f'Saving progress ({i+1}/{args.train_sims}).')
























