import os
import numpy as np


def queue_saver(queue, num):
	root = os.getcwd()
	try:
		os.mkdir('queue_data')
	except:
		pass

	os.chdir('queue_data')
	queue.save(num)
	os.chdir(root)


def policy_saver(policy, num):
	root = os.getcwd()
	try:
		os.mkdir('policies')
	except:
		pass

	os.chdir('policies')
	np.save(f'policy_{num}.npy', policy)
	os.chdir(root)

