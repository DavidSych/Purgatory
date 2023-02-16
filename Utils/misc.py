import os
import numpy as np


def my_chdir(dir_name):
	if not os.path.isdir(dir_name):
		os.mkdir(dir_name)
	os.chdir(dir_name)


def queue_saver(queue, num):
	root = os.getcwd()
	my_chdir('queue_data')
	queue.save(num)
	os.chdir(root)


def policy_saver(policy, num):
	root = os.getcwd()
	my_chdir('policies')
	np.save(f'policy_{num}', policy)
	os.chdir(root)



