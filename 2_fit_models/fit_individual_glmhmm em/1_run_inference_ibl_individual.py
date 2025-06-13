#  Fit GLM-HMM to data from all IBL animals together.  These fits will be
#  used to initialize the models for individual animals
import os
import sys

import autograd.numpy as np
from tqdm.contrib.concurrent import process_map

from glm_hmm_utils import load_cluster_arr, load_session_fold_lookup, \
    load_animal_list, load_data, create_violation_mask, \
    launch_glm_hmm_job

D = 1  # data (observations) dimension
C = 2  # number of output types/categories
N_em_iters = 300  # number of EM iterations


def fit(prior_sigma, transition_alpha, K, fold, iter):
    iter = int(iter)
    fold = int(fold)
    K = int(K)

    animal_list = load_animal_list(data_dir + 'animal_list.npz')

    for i, animal in enumerate(animal_list):
        if animal == "CSHL_008":
            animal_file = data_dir + animal + '_processed.npz'
            session_fold_lookup_table = load_session_fold_lookup(
                data_dir + animal + '_session_fold_lookup.npz')

            global_fit = False

            inpt, y, session = load_data(animal_file)
            #  append a column of ones to inpt to represent the bias covariate:
            inpt = np.hstack((inpt, np.ones((len(inpt), 1))))
            y = y.astype('int')

            overall_dir = results_dir + animal + '/'

            # Identify violations for exclusion:
            violation_idx = np.where(y == -1)[0]
            nonviolation_idx, mask = create_violation_mask(violation_idx,
                                                           inpt.shape[0])

            init_param_file = global_data_dir + \
                              'best_global_params/best_params_K_' + \
                              str(K) + '.npz'

            # create save directory for this initialization/fold combination:
            save_directory = overall_dir + '/GLM_HMM_K_' + str(
                K) + '/' + 'fold_' + str(fold) + '/' + '/iter_' + str(iter) + '/'
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)

            launch_glm_hmm_job(inpt, y, session, mask, session_fold_lookup_table,
                               K, D, C, N_em_iters, transition_alpha, prior_sigma,
                               fold, iter, global_fit, init_param_file,
                               save_directory)


if __name__ == '__main__':
    global_data_dir = '../../data/ibl/data_for_cluster/'
    data_dir = global_data_dir + 'data_by_animal/'
    results_dir = '../../results/ibl_individual_fit/'
    sys.path.insert(0, '../fit_global_glmhmm/')

    num_folds = 5

    # Load external files:
    cluster_arr_file = data_dir + 'cluster_job_arr.npz'
    # Load cluster array job parameters:
    cluster_arr = load_cluster_arr(cluster_arr_file)


    # for prior_sigma, transition_alpha, K, fold, iter in cluster_arr:
    #     fit(prior_sigma, transition_alpha, K, fold, iter)

    def helper(data):
        fit(data[0], data[1], data[2], data[3], data[4])


    process_map(
        helper,
        cluster_arr,
        max_workers=6,  # or whatever number of processes you want
        desc="fitting"
    )
