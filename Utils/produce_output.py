import matplotlib.pyplot as plt
import numpy as np
import pickle, os


def draw_all(algorithm, folder):
	root = os.getcwd()
	os.chdir(f'../Results/{algorithm}/{folder}')
	try:
		os.mkdir('figures')
	except:
		pass

	os.chdir('queue_data')
	args = pickle.load(open('args.pickle', 'rb'))
	for num in range(args.train_sims):
		leaving_time = np.load(f'leaving_time_{num}.npy')
		leaving_payment = np.load(f'leaving_payment_{num}.npy')

		fig, axs = plt.subplots(2, 1)
		fig.tight_layout(pad=1.5)
		axs[0].bar(np.arange(args.T), leaving_time / np.sum(leaving_time), align='center')
		axs[0].set_xlabel('Time')
		axs[0].set_ylabel('Percentage')
		axs[1].bar(np.arange(args.Q + 1), leaving_payment / np.sum(leaving_payment), align='center')
		axs[1].set_xlabel('Payment')
		axs[1].set_ylabel('Percentage')
		plt.savefig(os.pardir + f'/figures/counts_{num}.png')
		plt.savefig(os.pardir + f'/figures/counts_{num}.pdf')
		plt.clf()

		average_payment = np.sum(np.arange(args.Q + 1) * leaving_payment / np.sum(leaving_payment))
		print(f'Average payment: {np.round(average_payment, 2)}, {np.round(100 * average_payment / args.F, 2)}% of F.')

	os.chdir(root)


algorithm = 'PPO'
folder = '2022-10-05_10-41'
draw_all(algorithm, folder)




