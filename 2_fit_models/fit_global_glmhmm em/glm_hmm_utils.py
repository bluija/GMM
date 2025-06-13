# This code is implemented in glm_hmm_utils.py
import sys
import types

import autograd.numpy as np
import autograd.numpy.random as npr
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from ssm.util import ensure_args_are_lists, ensure_args_not_none, \
    ensure_slds_args_not_none, ensure_variational_args_are_lists, \
    replicate, collapse, ssm_pbar

import ssm

print("CUSTOM EM")


def load_data(animal_file):
    container = np.load(animal_file, allow_pickle=True)
    data = [container[key] for key in container]
    inpt = data[0]
    y = data[1]
    session = data[2]
    return inpt, y, session


def load_cluster_arr(cluster_arr_file):
    container = np.load(cluster_arr_file, allow_pickle=True)
    data = [container[key] for key in container]
    cluster_arr = data[0]
    return cluster_arr


def load_glm_vectors(glm_vectors_file):
    container = np.load(glm_vectors_file)
    data = [container[key] for key in container]
    loglikelihood_train = data[0]
    recovered_weights = data[1]
    return loglikelihood_train, recovered_weights


def load_global_params(global_params_file):
    container = np.load(global_params_file, allow_pickle=True)
    data = [container[key] for key in container]
    global_params = data[0]
    return global_params


def partition_data_by_session(inpt, y, mask, session):
    '''
    Partition inpt, y, mask by session
    :param inpt: arr of size TxM
    :param y:  arr of size T x D
    :param mask: Boolean arr of size T indicating if element is violation or
    not
    :param session: list of size T containing session ids
    :return: list of inpt arrays, data arrays and mask arrays, where the
    number of elements in list = number of sessions and each array size is
    number of trials in session
    '''
    inputs = []
    datas = []
    indexes = np.unique(session, return_index=True)[1]
    unique_sessions = [session[index] for index in sorted(indexes)]
    counter = 0
    masks = []
    for sess in unique_sessions:
        idx = np.where(session == sess)[0]
        counter += len(idx)
        inputs.append(inpt[idx, :])
        datas.append(y[idx, :])
        masks.append(mask[idx, :])
    assert counter == inpt.shape[0], "not all trials assigned to session!"
    return inputs, datas, masks


def load_session_fold_lookup(file_path):
    container = np.load(file_path, allow_pickle=True)
    data = [container[key] for key in container]
    session_fold_lookup_table = data[0]
    return session_fold_lookup_table


def load_animal_list(file):
    container = np.load(file, allow_pickle=True)
    data = [container[key] for key in container]
    animal_list = data[0]
    return animal_list


def launch_glm_hmm_job(inpt, y, session, mask, session_fold_lookup_table, K, D,
                       C, N_em_iters, transition_alpha, prior_sigma, fold,
                       iter, global_fit, init_param_file, save_directory):
    print("Starting inference with K = " + str(K) + "; Fold = " + str(fold) +
          "; Iter = " + str(iter))
    sys.stdout.flush()
    sessions_to_keep = session_fold_lookup_table[np.where(
        session_fold_lookup_table[:, 1] != fold), 0]
    idx_this_fold = [str(sess) in sessions_to_keep for sess in session]
    this_inpt, this_y, this_session, this_mask = inpt[idx_this_fold, :], \
        y[idx_this_fold, :], \
        session[idx_this_fold], \
        mask[idx_this_fold]
    # Only do this so that errors are avoided - these y values will not
    # actually be used for anything (due to violation mask)
    this_y[np.where(this_y == -1), :] = 1
    inputs, datas, masks = partition_data_by_session(
        this_inpt, this_y, this_mask, this_session)
    # Read in GLM fit if global_fit = True:
    if global_fit == True:
        _, params_for_initialization = load_glm_vectors(init_param_file)
    else:
        params_for_initialization = load_global_params(init_param_file)
    M = this_inpt.shape[1]
    npr.seed(iter)
    fit_glm_hmm(datas,
                inputs,
                masks,
                K,
                D,
                M,
                C,
                N_em_iters,
                transition_alpha,
                prior_sigma,
                global_fit,
                params_for_initialization,
                save_title=save_directory + 'glm_hmm_raw_parameters_itr_' +
                           str(iter) + '.npz')


def m_step(expectations, datas, inputs, regularizer=1.0):
    """
    M-step for GLM-HMM that calculates parameters from scratch.

    Parameters
    ----------
    expectations : list of tuples (Ez, Ezzp1, normalizer)
        - Ez     : (T, K) expected state responsibilities
        - Ezzp1  : (T, K, K) expected state transitions
        - normalizer : float (unused)

    datas : list of ndarray (T,)
        - Binary output labels (choices), one array per session

    inputs : list of ndarray (T, M)
        - Feature matrix (design matrix) for each session

    regularizer : float
        - L2 regularization coefficient (Gaussian prior on weights)

    Returns
    -------
    dict with keys:
        'pi' : ndarray (K,)
            - Updated initial state distribution
        'A' : ndarray (K, K)
            - Updated transition matrix
        'w' : ndarray (K * M,)
            - Flattened GLM weights for each state
    """
    # Infer K and M from inputs and expectations
    Ez_sample, Ezzp1_sample, _ = expectations[0]
    K = Ez_sample.shape[1]
    M = inputs[0].shape[1]

    # --- Update initial state distribution ---
    pi_numer = np.zeros(K)
    for Ez, _, _ in expectations:
        pi_numer += Ez[0]
    pi_new = pi_numer / pi_numer.sum()

    # --- Update transition matrix ---
    A_num = np.zeros((K, K))
    A_den = np.zeros(K)
    for _, Ezzp1, _ in expectations:
        E = Ezzp1.squeeze()
        A_num += E
        A_den += E.sum(axis=1)
    A_new = A_num / A_den[:, None]

    # --- Update GLM weights ---
    def neg_ECLL(w_flat):
        w = w_flat.reshape((K, M))
        loss = 0
        for (Ez, _, _), x, y in zip(expectations, inputs, datas):
            for k in range(K):
                logits = x @ w[k]
                log_prob = y * logits - np.log1p(np.exp(logits))
                loss += np.sum(Ez[:, k] * log_prob)
        loss -= 0.5 * regularizer * np.sum(w ** 2)
        return -loss

    def grad_neg_ECLL(w_flat):
        w = w_flat.reshape((K, M))
        grad = np.zeros_like(w)
        for (Ez, _, _), x, y in zip(expectations, inputs, datas):
            for k in range(K):
                logits = x @ w[k]  # (T,)
                probs = 1 / (1 + np.exp(-logits))  # (T,)
                # ensure Ez_k is a 1-D array of length T
                grad[k] += (Ez[:, k][:, None] * (y.squeeze() - probs)[:, None] * x).sum(axis=0)

        grad -= regularizer * w
        return -grad.flatten()

    # Initialize weights
    w0 = np.random.randn(K * M) * 0.1
    res = minimize(neg_ECLL, w0, jac=grad_neg_ECLL, method='BFGS')
    w_new = res.x

    return {'pi': pi_new, 'A': A_new, 'w': w_new}


# Bind custom functions
def fit_glm_hmm(datas, inputs, masks, K, D, M, C, N_em_iters,
                transition_alpha, prior_sigma, global_fit,
                params_for_initialization, save_title):
    '''
    Instantiate and fit GLM-HMM model
    :param datas:
    :param inputs:
    :param masks:
    :param K:
    :param D:
    :param M:
    :param C:
    :param N_em_iters:
    :param global_fit:
    :param glm_vectors:
    :param save_title:
    :return:
    '''
    if global_fit == True:
        # Prior variables
        # Choice of prior
        this_hmm = ssm.HMM(K,
                           D,
                           M,
                           observations="input_driven_obs",
                           observation_kwargs=dict(C=C,
                                                   prior_sigma=prior_sigma),
                           transitions="sticky",
                           transition_kwargs=dict(alpha=transition_alpha,
                                                  kappa=0))
        # Initialize observation weights as GLM weights with some noise:
        glm_vectors_repeated = np.tile(params_for_initialization, (K, 1, 1))
        glm_vectors_with_noise = glm_vectors_repeated + np.random.normal(
            0, 0.2, glm_vectors_repeated.shape)
        this_hmm.observations.params = glm_vectors_with_noise
    else:
        # Choice of prior
        this_hmm = ssm.HMM(K,
                           D,
                           M,
                           observations="input_driven_obs",
                           observation_kwargs=dict(C=C,
                                                   prior_sigma=prior_sigma),
                           transitions="sticky",
                           transition_kwargs=dict(alpha=transition_alpha,
                                                  kappa=0))
        # Initialize HMM-GLM with global parameters:
        this_hmm.params = params_for_initialization
        # Get log_prior of transitions:
    print("=== fitting GLM-HMM ========")
    sys.stdout.flush()

    # # Bind custom expected_states and _fit_em to this_hmm
    this_hmm.expected_states = types.MethodType(expected_states, this_hmm)
    this_hmm._fit_em = types.MethodType(_fit_em, this_hmm)

    # Fit this HMM and calculate marginal likelihood
    lls = this_hmm.fit(datas,
                       inputs=inputs,
                       masks=masks,
                       method="em",
                       num_iters=N_em_iters,
                       initialize=False,
                       tolerance=10 ** -4)
    # Save raw parameters of HMM, as well as loglikelihood during training
    np.savez(save_title, this_hmm.params, lls)
    return None


def expected_states(self, data, input=None, mask=None, tag=None):
    """
    Compute expected states using the forward-backward algorithm.

    Parameters:
    - data: (T,) array of binary observations (0 or 1)
    - input: (T, D) array of inputs
    - mask: (T,) array of 1s (valid) or 0s (invalid)
    - tag: optional tag for multiple sequences

    Returns:
    - gamma: (T, K) array of posterior probabilities P(z_t | y)
    - xi: (T-1, K, K) or (1, K, K) array of joint posterior probabilities P(z_t, z_{t+1} | y)
    - loglik: scalar log likelihood
    """
    T = len(data)
    pi0 = self.init_state_distn.initial_state_distn
    Ps = self.transitions.transition_matrices(data, input, mask, tag)
    log_likes = self.observations.log_likelihoods(data, input, mask, tag)

    # Check if transition matrix is stationary
    stationary = Ps.shape[0] == 1

    # Compute log transition matrices
    with np.errstate(divide="ignore"):
        log_Ps = np.log(Ps + 1e-10)

    # Forward pass
    alpha = np.zeros((T, self.K))
    log_p_y = np.zeros(T)
    alpha[0] = np.log(pi0 + 1e-10) + log_likes[0]
    log_p_y[0] = logsumexp(alpha[0])

    # Commented forward pass normalizer
    # alpha[0] -= log_p_y[0]

    for t in range(1, T):
        for k in range(self.K):
            alpha[t, k] = log_likes[t, k] + logsumexp(alpha[t - 1] + log_Ps[min(t - 1, log_Ps.shape[0] - 1), :, k])
        log_p_y[t] = logsumexp(alpha[t])

        # Commented forward pass normalizer
        # alpha[t] -= log_p_y[t]

    # Backward pass
    beta = np.zeros((T, self.K))
    beta[-1] = 0
    for t in range(T - 2, -1, -1):
        for k in range(self.K):
            beta[t, k] = logsumexp(log_Ps[min(t, log_Ps.shape[0] - 1), k, :] + log_likes[t + 1, :] + beta[t + 1, :])

    # Compute gamma (expected_states)
    gamma = alpha + beta
    gamma -= logsumexp(gamma, axis=1, keepdims=True)
    gamma = np.exp(gamma)

    # Compute xi (expected_joints)
    if stationary:
        xi = np.zeros((1, self.K, self.K))
        log_xi = np.zeros((1, self.K, self.K))
        for t in range(T - 1):
            for i in range(self.K):
                for j in range(self.K):
                    log_xi[0, i, j] += np.exp(
                        alpha[t, i] + log_Ps[0, i, j] + log_likes[t + 1, j] + beta[t + 1, j] - log_p_y[-1])
        xi[0] = log_xi[0] / np.sum(log_xi[0])  # Normalize
    else:
        xi = np.zeros((T - 1, self.K, self.K))
        for t in range(T - 1):
            for i in range(self.K):
                for j in range(self.K):
                    xi[t, i, j] = alpha[t, i] + log_Ps[t, i, j] + log_likes[t + 1, j] + beta[t + 1, j]
            xi[t] -= logsumexp(xi[t])  # Normalize
            xi[t] = np.exp(xi[t])  # Convert to probability

    # log-likelihood
    loglik = log_p_y[-1]  # Normalizer from forward pass

    return gamma, xi, loglik


# The original code.
def _fit_em(self, datas, inputs, masks, tags, verbose=2, num_iters=100, tolerance=0,
            init_state_mstep_kwargs={}, transitions_mstep_kwargs={}, observations_mstep_kwargs={}, **kwargs):
    """
    Fit the parameters with expectation maximization.

    Parameters:
    - datas: list of observation arrays
    - inputs: list of input arrays
    - masks: list of mask arrays
    - tags: list of tags
    - verbose: verbosity level
    - num_iters: maximum number of EM iterations
    - tolerance: convergence tolerance
    """
    lls = [self.log_probability(datas, inputs, masks, tags)]
    pbar = ssm_pbar(num_iters, verbose, "LP: {:.1f}", [lls[-1]])

    for itr in pbar:
        # E step: compute expected latent states with current parameters
        expectations = [self.expected_states(data, input, mask, tag)
                        for data, input, mask, tag
                        in zip(datas, inputs, masks, tags)]

        # M step: maximize expected log joint wrt parameters

        mstep_results = m_step(expectations, datas, inputs, regularizer=1.0)

        self.init_state_distn.log_pi0 = np.log(mstep_results['pi'] + 1e-8)

        self.transitions.log_Ps = np.log(mstep_results['A'] + 1e-8)

        self.observations.w = mstep_results['w'].reshape(self.K, -1)

        # self.init_state_distn.m_step(expectations, datas, inputs, masks, tags, **init_state_mstep_kwargs)
        # self.transitions.m_step(expectations, datas, inputs, masks, tags, **transitions_mstep_kwargs)
        # self.observations.m_step(expectations, datas, inputs, masks, tags, **observations_mstep_kwargs)

        # Store progress
        lls.append(self.log_prior() + sum([ll for (_, _, ll) in expectations]))

        if verbose == 2:
            pbar.set_description("LP: {:.1f}".format(lls[-1]))

        # Check for convergence
        if itr > 0 and abs(lls[-1] - lls[-2]) < tolerance:
            if verbose == 2:
                pbar.set_description("Converged to LP: {:.1f}".format(lls[-1]))
            break

    return lls


def create_violation_mask(violation_idx, T):
    """
    Return indices of nonviolations and also a Boolean mask for inclusion (1
    = nonviolation; 0 = violation)
    :param test_idx:
    :param T:
    :return:
    """
    mask = np.array([i not in violation_idx for i in range(T)])
    nonviolation_idx = np.arange(T)[mask]
    mask = mask + 0
    assert len(nonviolation_idx) + len(
        violation_idx
    ) == T, "violation and non-violation idx do not include all dta!"
    return nonviolation_idx, np.expand_dims(mask, axis=1)
