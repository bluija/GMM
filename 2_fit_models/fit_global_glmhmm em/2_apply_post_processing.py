# Create a matrix of size num_models x num_folds containing
# normalized loglikelihood for both train and test splits
import json

import numpy as np

from post_processing_utils import load_data, load_session_fold_lookup, \
    prepare_data_for_cv, calculate_baseline_test_ll, \
    return_glmhmm_nll

if __name__ == '__main__':
    data_dir = '../../data/ibl/data_for_cluster/'
    results_dir = '../../results/ibl_global_fit/'

    # Load data
    inpt, y, session = load_data(data_dir + 'all_animals_concat.npz')
    session_fold_lookup_table = load_session_fold_lookup(
        data_dir + 'all_animals_concat_session_fold_lookup.npz')

    # Parameters
    C = 2  # number of output classes
    # num_folds = 5  # number of folds
    num_folds = 2  # number of folds
    D = 1  # number of output dimensions
    # K_max = 4  # maximum number of latent states
    num_models = 3  # model for each latent + 2 lapse models

    animal_preferred_model_dict = {}
    models = ["GLM_HMM"]

    cvbt_folds_model = np.zeros((num_models, num_folds))
    cvbt_train_folds_model = np.zeros((num_models, num_folds))

    # Save best initialization for each model-fold combination
    best_init_cvbt_dict = {}
    for fold in range(num_folds):
        test_inpt, test_y, test_nonviolation_mask, this_test_session, \
            train_inpt, train_y, train_nonviolation_mask, this_train_session, M, \
            n_test, n_train = prepare_data_for_cv(
            inpt, y, session, session_fold_lookup_table, fold)
        ll0 = calculate_baseline_test_ll(
            train_y[train_nonviolation_mask == 1, :],
            test_y[test_nonviolation_mask == 1, :], C)
        ll0_train = calculate_baseline_test_ll(
            train_y[train_nonviolation_mask == 1, :],
            train_y[train_nonviolation_mask == 1, :], C)
        for model in models:
            print("model = " + str(model))
            if model == "GLM_HMM":
                for K in [2, 3, 4]:
                    print("K = " + str(K))
                    model_idx = K - 2
                    cvbt_folds_model[model_idx, fold], \
                        cvbt_train_folds_model[
                            model_idx, fold], _, _, init_ordering_by_train = \
                        return_glmhmm_nll(
                            np.hstack((inpt, np.ones((len(inpt), 1)))), y,
                            session, session_fold_lookup_table, fold,
                            K, D, C, results_dir)
                    # Save best initialization to dictionary for later:
                    key_for_dict = '/GLM_HMM_K_' + str(K) + '/fold_' + str(
                        fold)
                    best_init_cvbt_dict[key_for_dict] = int(
                        init_ordering_by_train[0])
    # Save best initialization directories across animals, folds and models
    # (only GLM-HMM):
    print(cvbt_folds_model)
    print(cvbt_train_folds_model)
    json_dump = json.dumps(best_init_cvbt_dict)
    f = open(results_dir + "/best_init_cvbt_dict.json", "w")
    f.write(json_dump)
    f.close()
    # Save cvbt_folds_model as numpy array for easy parsing across all
    # models and folds
    np.savez(results_dir + "/cvbt_folds_model.npz", cvbt_folds_model)
    np.savez(results_dir + "/cvbt_train_folds_model.npz",
             cvbt_train_folds_model)
