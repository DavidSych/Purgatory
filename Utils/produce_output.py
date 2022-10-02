import matplotlib.pyplot as plt
import numpy as np
import pickle, os


def draw(num):
	root = os.getcwd()
	os.chdir('../Results')
	args = pickle.load(open('args.pickle', 'rb'))
	leaving_time = np.load(f'leaving_time_{num}.npy')
	leaving_payment = np.load(f'leaving_payment_{num}.npy')

	fig, axs = plt.subplots(2, 1)
	fig.tight_layout(pad=1.5)
	axs[0].hist(np.arange(args.T), weights=leaving_time / np.sum(leaving_time))
	axs[0].set_xlabel('Time')
	axs[0].set_ylabel('Percentage')
	axs[1].hist(np.arange(args.F + args.Q), weights=leaving_payment / np.sum(leaving_payment))
	axs[1].set_xlabel('Payment')
	axs[1].set_ylabel('Percentage')
	plt.savefig('counts.png')

	os.chdir(root)

draw(0)



