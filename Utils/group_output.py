import matplotlib.pyplot as plt
import numpy as np
import pickle, os
from Utils.misc import my_chdir
import matplotlib as mpl

mpl.rcParams.update({
   'axes.labelsize': 10,
   'font.size': 10,
   'legend.fontsize': 10,
   'xtick.labelsize': 8,
   'ytick.labelsize': 8,
   'figure.figsize': [5.5, 4.5]
})


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
	:return: None
	'''
	root = os.getcwd()
	y = np.empty(len(folders))
	for i, folder in enumerate(folders):
		os.chdir(f'../Results/Group/{folder}')
		args = pickle.load(open('args.pickle', 'rb'))
		alg_id = np.where(np.array(args.algorithms) == algorithm)[0]
		os.chdir('queue_data')
		train_sims = args.train_sims - 1
		leaving_payment = np.load(f'leaving_payment_{train_sims}.npy')
		info = pickle.load(open(f'info_{train_sims}.pickle', 'rb'))

		total_payment = np.sum(np.arange(args.Q + args.F) * leaving_payment[alg_id])
		min_payment = args.Q * args.k / args.x_mean
		payment = total_payment - min_payment
		y[i] = payment / info['w']

		os.chdir(root)

	os.chdir(f'../Results/Group')
	plt.plot(plot_info['x'], y)
	plt.xlabel(plot_info['x label'])
	plt.ylabel(plot_info['y label'])
	plt.title(plot_info['title'])
	plt.savefig('group_output.pdf')
	np.save(f"{plot_info['title']}.npy", y)
	os.chdir(root)


algorithm = 'ppo1'
folders = ['2023-02-14_15-17'] #[f'_ppo_F4_Q6_T4_k2_x{x}_p0.5' for x in range(10, 101, 10)]
plot_info = {'x': 10 * (np.arange(len(folders)) + 1),
			 'x label': 'Number of incomers ($x$)',
			 'y label': 'Average Payment above Minimum',
			 'title': ''}

prepare_folders('Group', folders)
draw_average_payment(algorithm, folders, plot_info)




