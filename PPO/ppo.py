import numpy as np
from Simulations.Environment.queue import Queue
import argparse
import datetime
import shutil
import pickle
import matplotlib.pyplot as plt

import cProfile
import os

#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

parser = argparse.ArgumentParser()
# TF params
parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility.")
parser.add_argument("--threads", default=15, type=int, help="Number of CPU threads to use.")
parser.add_argument("--hidden_layer_actor", default=4, type=int, help="Size of the hidden layer of the network.")
parser.add_argument("--hidden_layer_critic", default=32, type=int, help="Size of the hidden layer of the network.")
parser.add_argument("--learning_rate", default=3e-4, type=float, help="Learning rate.")
parser.add_argument("--entropy_weight", default=1e-2, type=float, help="Entropy regularization constant.")
parser.add_argument("--baseline_weight", default=3., type=float, help="Value loss scaling, i.e. critic learning rate scaling.")
parser.add_argument("--l2_weight", default=1e-2, type=float, help="L2 regularization constant.")
parser.add_argument("--clip_norm", default=0.1, type=float, help="Gradient clip norm.")

parser.add_argument("--buffer_len", default=20_000, type=int, help="Number of time steps to train on.")
parser.add_argument("--epsilon", default=0.05, type=float, help="Clipping constant.")
parser.add_argument("--gamma", default=1, type=float, help="Return discounting.")
parser.add_argument("--_lambda", default=0.97, type=float, help="Advantage discounting.")
parser.add_argument("--train_cycles", default=64, type=int, help="Number of PPO passes.")
parser.add_argument("--train_sims", default=128, type=int, help="How many simulations to train from.")
parser.add_argument("--evaluate", default=False, type=bool, help="If NashConv should be computed as well.")

# Queue parameters
parser.add_argument("--F", default=4, type=int, help="End fine to pay.")
parser.add_argument("--Q", default=4, type=int, help="End fine to pay.")
parser.add_argument("--fined_penalty", default=0, type=int, help="Additional penalty for leaving due to a fine.")
parser.add_argument("--T", default=4, type=int, help="Time to survive in queue.")
parser.add_argument("--k", default=5, type=int, help="How many people have to pay in each step.")
parser.add_argument("--g", default=1, type=int, help="How many groups to use.")
parser.add_argument("--tau", default=1, type=int, help="Don't use agents added to queue before tau * T steps.")
parser.add_argument("--x_mean", default=100, type=float, help="Mean number of agents to add each step.")
parser.add_argument("--x_std", default=5, type=float, help="Standard deviation of the number of agents to add each step.")
parser.add_argument("--N_init", default=100, type=int, help="Initial number of agents.")
parser.add_argument("--ignorance_distribution", default='uniform', type=str, help="What distribuin to use to sample probability of ignorance.")
parser.add_argument("--p_min", default=0.7, type=float, help="Parameter of uniform distribution of ignorance.")
parser.add_argument("--alpha", default=2, type=float, help="Parameter of Beta distribution of ignorance.")
parser.add_argument("--beta", default=4, type=float, help="Parameter of Beta distribution of ignorance.")
parser.add_argument("--reward_shaping", default=False, type=bool, help="If rewards shaping should be used.")

args = parser.parse_args([] if "__file__" not in globals() else None)

np.random.seed(args.seed)

queue = Queue(args)
queue.initialize()
for _ in range(100):
	queue.step(np.zeros(len(queue.agents)))

























