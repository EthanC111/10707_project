import numpy as np
import matplotlib.pyplot as plt


plt.style.use('seaborn')
step = 25
fig, axs = plt.subplots(1, 4, figsize=(30,5))
for i, l in enumerate([5, 20, 50, 100]):

	train_trajectory = np.load("cache/models/t5-v1_1-small-rte-" + str(l) + '/train_loss.npy')[:step]
	valid_trajectory = np.load("cache/models/t5-v1_1-small-rte-" + str(l) + '/eval_loss.npy')[:step]
	axs[i].plot(100 * np.arange(len(train_trajectory)), train_trajectory, label = 'train', color='blue')
	axs[i].plot(100 * np.arange(len(train_trajectory)), valid_trajectory, label = 'evaluation', color='orange')

	axs[i].set_ylabel('CE Loss', fontsize = 25)
	axs[i].set_xlabel('Training Step', fontsize = 25)
	axs[i].set_title('Prompt Length = '+str(l), fontsize = 25)
	axs[i].legend(fontsize = 25,loc='upper right')

plt.savefig('loss.png', bbox_inches='tight')
