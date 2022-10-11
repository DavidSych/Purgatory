import matplotlib.pyplot as plt
import numpy as np
import pickle, os
from Simulations.Utils.misc import my_chdir


def prepare_folders(algorithm, folder):
	root = os.getcwd()
	os.chdir(f'../Results/{algorithm}/{folder}')
	my_chdir('figures')
	my_chdir('counts')
	os.chdir(root)


def draw_final_policy(algorithm, folder):
	root = os.getcwd()
	os.chdir(f'../Results/{algorithm}/{folder}')
	args = pickle.load(open('args.pickle', 'rb'))
	os.chdir('policies')
	final_policy = np.load(f'policy_{len(os.listdir(os.getcwd())) - 1}.npy')
	os.chdir('../figures')
	for i, feature in enumerate(['Payment', 'Time', 'Position']):
		avr_axes = [0, 1, 2]
		avr_axes.pop(i)
		avr_policy = np.mean(final_policy, axis=tuple(avr_axes))
		avr_percentage = np.round(100 * avr_policy, decimals=1)
		x = np.arange(avr_policy.shape[0])
		plt.grid(0.25)
		for action in range(1, args.F + 1):
			plt.plot(x, avr_percentage[:, action], label=f'Pay {action}')

		plt.xlabel(feature)
		plt.ylabel('Probability [%]')
		plt.title(f'Policy as a function of {feature}')
		plt.ylim(ymin=0)
		plt.legend()
		plt.savefig(f'{feature.lower()}_averaged.pdf')
		plt.clf()

	os.chdir(root)


def draw_leaving_counts(algorithm, folder):
	root = os.getcwd()
	os.chdir(f'../Results/{algorithm}/{folder}/queue_data')
	args = pickle.load(open('args.pickle', 'rb'))
	for num in range(len(os.listdir(os.getcwd())) // 2):
		leaving_time = np.load(f'leaving_time_{num}.npy')
		leaving_payment = np.load(f'leaving_payment_{num}.npy')

		fig, axs = plt.subplots(2, 1)
		fig.tight_layout(pad=1.5)
		axs[0].bar(np.arange(args.T) + 1, leaving_time / np.sum(leaving_time), align='center')
		axs[0].set_xlabel('Time')
		axs[0].set_ylabel('Percentage')
		axs[1].bar(np.arange(args.Q + 1), leaving_payment / np.sum(leaving_payment), align='center')
		axs[1].set_xlabel('Payment')
		axs[1].set_ylabel('Percentage')
		plt.savefig(os.pardir + f'/figures/counts/counts_{num}.pdf')
		plt.clf()

		average_payment = np.sum(np.arange(args.Q + 1) * leaving_payment / np.sum(leaving_payment))
		min_payment = args.Q * args.k / args.x_mean
		payment = average_payment - min_payment
		print(f'Average payment above minimum: {np.round(payment, 2)}, {np.round(100 * payment / args.F, 2)}% of F.')

	os.chdir(root)


algorithm = 'PW'
folder = '2022-10-10_21-55'
prepare_folders(algorithm, folder)
draw_final_policy(algorithm, folder)
draw_leaving_counts(algorithm, folder)




