import argparse
import numpy as np
from scipy.special import softmax
from numpy import linalg as LA
import torch
import os
########################################################################################################################
#  Calculate Importance
########################################################################################################################

# Define and parse command line arguments
parser = argparse.ArgumentParser(description='Calculate sample-wise importance',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dynamics_path', type=str, default='./checkpoint/all-dataset/npy/',
                    help='Folder to save dynamics.')
parser.add_argument('--mask_path', type=str, default='./checkpoint/generated_mask/',
                    help='Folder to save mask.')
parser.add_argument('--trajectory_len', default=10, type=int,
                    help='Length of the trajectory.')
parser.add_argument('--window_size', default=10, type=int,
                    help='Size of the sliding window.')
parser.add_argument('--decay', default=0.9, type=float,
                    help='Decay factor for moving average.')

args = parser.parse_args()

def generate(probs, losses, indexes):
    # Initialize variables
    k = 0
    window_size = args.window_size
    moving_averages = []

    # Iterate through the trajectory
    while k < args.trajectory_len - window_size + 1:
        probs_window = probs[k: k + window_size, :]
        indexes_window = indexes[k: k + window_size, :]
        probs_window_softmax = softmax(probs_window, axis=2)
        
        probs_window_rere = []
        # Reorganize probabilities according to indexes
        for i in range(window_size):
            probs_window_re = torch.zeros_like(torch.tensor(probs_window_softmax[0, :, :]))
            probs_window_re = probs_window_re.index_add(0, torch.tensor(indexes_window[i], dtype=int),
                                                        torch.tensor(probs_window_softmax[i, :]))
            probs_window_rere.append(probs_window_re)
       
        probs_window_kd = []
        # Calculate KL divergence in one window
        for j in range(window_size - 1):
            log = torch.log(probs_window_rere[j + 1] + 1e-8) - torch.log(probs_window_rere[j] + 1e-8)
            kd = torch.abs(torch.multiply(probs_window_rere[j + 1], log)).sum(axis=1)
            probs_window_kd.append(kd)
        probs_window_kd = np.array(probs_window_kd)

        window_average = probs_window_kd.sum(0) / (window_size - 1)
        
        window_diffdiff = []
        for ii in range(window_size - 1):
            window_diffdiff.append((np.array(probs_window_kd[ii]) - window_average))
        window_diffdiff_norm = LA.norm(np.array(window_diffdiff), axis=0) 
        moving_averages.append(window_diffdiff_norm * args.decay * (1 - args.decay) ** (args.trajectory_len - window_size - k))
        k += 1
        print(str(k) + ' window ok!')

    moving_averages_sum = np.squeeze(sum(np.array(moving_averages), 0))
    data_mask = moving_averages_sum.argsort() 
    moving_averages_sum_sort = np.sort(moving_averages_sum) 

    # Save the generated mask and scores
    if not os.path.exists(args.mask_path):
        os.makedirs(args.mask_path)
    
    np.save(args.mask_path + 'data_mask_win' + str(args.window_size) + '_ep' + str(args.trajectory_len) + '.npy', np.array(data_mask))
    np.save(args.mask_path + 'score_win' + str(args.window_size) + '_ep' + str(args.trajectory_len) + '.npy', np.array(moving_averages_sum_sort))

if __name__ == '__main__':
    # Load sample probabilities, losses, and indexes
    probs = []
    losses = []
    indexes = []
    for i in range(args.trajectory_len):
        probs.append(np.load(args.dynamics_path + str(i) + '_Output.npy'))
        losses.append(np.load(args.dynamics_path + str(i) + '_Loss.npy'))
        indexes.append(np.load(args.dynamics_path + str(i) + '_Index.npy'))

    probs = np.array(probs)
    losses = np.array(losses)
    indexes = np.array(indexes)

    generate(probs=probs, losses=losses, indexes=indexes)
