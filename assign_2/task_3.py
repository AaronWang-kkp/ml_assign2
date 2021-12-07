import numpy as np
import os
import matplotlib.pyplot as plt
from print_values import *
from plot_data_all_phonemes import *
from plot_data import *
import random
from sklearn.preprocessing import normalize
from get_predictions import *
from plot_gaussians import *

# File that contains the data
data_npy_file = 'data/PB_data.npy'

# Loading data from .npy file
data = np.load(data_npy_file, allow_pickle=True)
data = np.ndarray.tolist(data)

# Make a folder to save the figures
figures_folder = os.path.join(os.getcwd(), 'figures')
if not os.path.exists(figures_folder):
    os.makedirs(figures_folder, exist_ok=True)

# Array that contains the phoneme ID (1-10) of each sample
phoneme_id = data['phoneme_id']
# frequencies f1 and f2
f1 = data['f1']
f2 = data['f2']

# Initialize array containing f1 & f2, of all phonemes.
X_full = np.zeros((len(f1), 2))
#########################################
# Write your code here
# Store f1 in the first column of X_full, and f2 in the second column of X_full
for i in range(len(f1)):
    X_full[i, 0] = f1[i]
    X_full[i, 1] = f2[i]
########################################/
X_full = X_full.astype(np.float32)

# number of GMM components
k = 6

#########################################
# Write your code here
X_phoneme_1 = np.zeros((np.sum(phoneme_id==1), 2))
a = np.argwhere(phoneme_id==1)
for i in range(len(X_phoneme_1)):
    X_phoneme_1[i] = X_full[a[i]]

X_phoneme_2 = np.zeros((np.sum(phoneme_id==2), 2))
b = np.argwhere(phoneme_id==2)
for i in range(len(X_phoneme_2)):
    X_phoneme_2[i] = X_full[b[i]]

X_phonemes_1_2 = np.vstack((X_phoneme_1,X_phoneme_2))

# Create an array named "X_phonemes_1_2", containing only samples that belong to phoneme 1 and samples that belong to phoneme 2.
# The shape of X_phonemes_1_2 will be two-dimensional. Each row will represent a sample of the dataset, and each column will represent a feature (e.g. f1 or f2)
# Fill X_phonemes_1_2 with the samples of X_full that belong to the chosen phonemes
# To fill X_phonemes_1_2, you can leverage the phoneme_id array, that contains the ID of each sample of X_full

# X_phonemes_1_2 = ...

########################################/

# Plot array containing the chosen phonemes

# Create a figure and a subplot
fig, ax1 = plt.subplots()

title_string = 'Phoneme 1 & 2'
# plot the samples of the dataset, belonging to the chosen phoneme (f1 & f2, phoneme 1 & 2)
plot_data(X=X_phonemes_1_2, title_string=title_string, ax=ax1)
# save the plotted points of phoneme 1 as a figure
plot_filename = os.path.join(os.getcwd(), 'figures', 'dataset_phonemes_1_2.png')
plt.savefig(plot_filename)


#########################################
# Write your code here
# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 1
# Get predictions on samples from both phonemes 1 and 2, from a GMM with k components, pretrained on phoneme 2
# Compare these predictions for each sample of the dataset, and calculate the accuracy, and store it in a scalar variable named "accuracy"

data_Theta_p1_k3 = 'data/GMM_params_phoneme_01_k_03.npy'
data_Theta_p1_k6 = 'data/GMM_params_phoneme_01_k_06.npy'
data_Theta_p2_k3 = 'data/GMM_params_phoneme_02_k_03.npy'
data_Theta_p2_k6 = 'data/GMM_params_phoneme_02_k_06.npy'
data_Theta1 = np.load(data_Theta_p1_k3, allow_pickle=True)
data_Theta1 = np.ndarray.tolist(data_Theta1)
data_Theta2 = np.load(data_Theta_p2_k3, allow_pickle=True)
data_Theta2 = np.ndarray.tolist(data_Theta2)
data_Theta3 = np.load(data_Theta_p1_k6, allow_pickle=True)
data_Theta3 = np.ndarray.tolist(data_Theta3)
data_Theta4 = np.load(data_Theta_p2_k6, allow_pickle=True)
data_Theta4 = np.ndarray.tolist(data_Theta4)

mu1 = data_Theta1['mu']
s1 = data_Theta1['s']
p1 = data_Theta1['p']

mu2 = data_Theta2['mu']
s2 = data_Theta2['s']
p2 = data_Theta2['p']

mu3 = data_Theta3['mu']
s3 = data_Theta3['s']
p3 = data_Theta3['p']

mu4 = data_Theta4['mu']
s4 = data_Theta4['s']
p4 = data_Theta4['p']

X = X_phonemes_1_2.copy()

Z1 = get_predictions(mu1, s1, p1, X)
Z2 = get_predictions(mu2, s2, p2, X)
Z3 = get_predictions(mu3, s3, p3, X)
Z4 = get_predictions(mu4, s4, p4, X)


A1 = 0
B1 = 0

for i in range(int(X.shape[0]/2)):
    if k ==3:
        if np.sum(Z1[i]) > np.sum(Z2[i]):
            A1 += 1
    if k ==6:
        if np.sum(Z3[i]) > np.sum(Z4[i]):
            A1 += 1
for i in range(int(X.shape[0]/2),int(X.shape[0])):
    if k == 3:
        if np.sum(Z2[i]) > np.sum(Z1[i]):
            B1 += 1
    if k == 6:
        if np.sum(Z4[i]) > np.sum(Z3[i]):
            B1 += 1

accuracy = ((A1+B1)/X.shape[0])*100
########################################/

print('Accuracy using GMMs with {} components: {:.2f}%'.format(k, accuracy))

################################################
# enter non-interactive mode of matplotlib, to keep figures open
plt.ioff()
plt.show()