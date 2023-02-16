import matplotlib.pyplot as plt
import numpy as np
import pickle, os
from Utils.misc import my_chdir


def prepare_folders(algorithm, folders):
	root = os.getcwd()
	for folder in folders:
		os.chdir(f'../Results/{algorithm}/{folder}')
		my_chdir('figures')
		my_chdir('counts')
		os.chdir(root)


def draw_average_payment(algorithm, folders, plot_info):
	'''
	Plots average of the leaving payments of agents above minimum, i.e. without kQ per step
	:param algorithm: str from [PPO]
	:param folders: list of folders with data
	:param info: dict including
				- x range and label
				- y label
	:return:
	'''
	root = os.getcwd()
	y = np.empty(len(folders))
	for i, folder in enumerate(folders):
		os.chdir(f'../Results/{algorithm}/{folder}')
		args = pickle.load(open('args.pickle', 'rb'))
		os.chdir('queue_data')
		train_sims = args.train_sims - 1
		leaving_payment = np.load(f'leaving_payment_{train_sims}.npy')
		info = pickle.load(open(f'info_{train_sims}.pickle', 'rb'))

		total_payment = np.sum(np.arange(args.Q + args.F) * leaving_payment)
		min_payment = args.Q * args.k / args.x_mean
		payment = total_payment - min_payment
		y[i] = payment / info['w']

		os.chdir(root)

	os.chdir(f'../Results/{algorithm}')
	plt.plot(plot_info['x'], y)
	plt.xlabel(plot_info['x label'])
	plt.ylabel(plot_info['y label'])
	plt.savefig('group_output.pdf')
	os.chdir(root)


algorithm = 'PPO'
folders = [f'_ppo_F4_Q6_T4_k2_x{x}_p0.5' for x in range(10, 101, 10)]
plot_info = {'x': 10 * (np.arange(len(folders)) + 1),
			 'x label': 'Number of incomers ($x$)',
			 'y label': 'Average Payment above Minimum'}

prepare_folders(algorithm, folders)
draw_average_payment(algorithm, folders, plot_info)




