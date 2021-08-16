""" gp.py

Bayesian optimisation of loss functions.
"""

import numpy as np
import sklearn.gaussian_process as gp

from scipy.stats import norm
from scipy.optimize import minimize
import random                           

def random_pram_sample(pram_dict): # for random hyper pram optimization
    sample = {}
    for pram in pram_dict:
        value = random.choice(pram_dict[pram])
        sample[pram] = value
    return sample


def expected_improvement(x, gaussian_process, evaluated_loss, greater_is_better=False, n_params=1):
    """ expected_improvement

    Expected improvement acquisition function.

    Arguments:
    ----------
        x: array-like, shape = [n_samples, n_hyperparams]
            The point for which the expected improvement needs to be computed.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: Numpy array.
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        n_params: int.
            Dimension of the hyperparameter space.

    """

    x_to_predict = x.reshape(-1, n_params)

    mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)

    if greater_is_better:
        loss_optimum = np.max(evaluated_loss)
    else:
        loss_optimum = np.min(evaluated_loss)

    scaling_factor = (-1) ** (not greater_is_better)

    # In case sigma equals zero
    with np.errstate(divide='ignore'):
        Z = scaling_factor * (mu - loss_optimum) / sigma
        expected_improvement = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
        expected_improvement[sigma == 0.0] == 0.0

    return -1 * expected_improvement


def sample_next_hyperparameter(acquisition_func, gaussian_process, evaluated_loss, greater_is_better=False,
                               bounds=(0, 10), n_restarts=25):
    """ sample_next_hyperparameter

    Proposes the next hyperparameter to sample the loss function for.

    Arguments:
    ----------
        acquisition_func: function.
            Acquisition function to optimise.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: array-like, shape = [n_obs,]
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        bounds: Tuple.
            Bounds for the L-BFGS optimiser.
        n_restarts: integer.
            Number of times to run the minimiser with different starting points.

    """
    best_x = None
    best_acquisition_value = 1
    n_params = bounds.shape[0]

    for starting_point in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, n_params)):

        res = minimize(fun=acquisition_func,
                       x0=starting_point.reshape(1, -1),
                       bounds=bounds,
                       method='L-BFGS-B',
                       args=(gaussian_process, evaluated_loss, greater_is_better, n_params))

        if res.fun < best_acquisition_value:
            best_acquisition_value = res.fun
            best_x = res.x
            # print(">>>> new opt params ", best_x)
    return best_x


def bayesian_optimisation(n_iters, x_train, y_train, x_test, y_test, sample_loss, bounds, d_param, x0,
                          gp_params=None, alpha=1e-5, epsilon=1e-7, seed=42):
    """ bayesian_optimisation

    Uses Gaussian Processes to optimise the loss function `sample_loss`.

    Arguments:
    ----------
        n_iters: integer.
            Number of iterations to run the search algorithm.
        sample_loss: function.
            Function to be optimised.
        bounds: array-like, shape = [n_params, 2].
            Lower and upper bounds on the parameters of the function `sample_loss`.
        x0: array-like, shape = [n_pre_samples, n_params].
            Array of initial points to sample the loss function for. If None, randomly
            samples from the loss function.
        n_pre_samples: integer.
            If x0 is None, samples `n_pre_samples` initial points from the loss function.
        gp_params: dictionary.
            Dictionary of parameters to pass on to the underlying Gaussian Process.
        random_search: integer.
            Flag that indicates whether to perform random search or L-BFGS-B optimisation
            over the acquisition function.
        alpha: double.
            Variance of the error term of the GP.
        epsilon: double.
            Precision tolerance for floats.
    """

    # convert dict to bounds array

    array_bounds = []
    for single_param in bounds:
        min_value = min(bounds[single_param])
        max_value = max(bounds[single_param])
        value = [min_value, max_value]
        array_bounds.append(value)
    array_bounds = np.array(array_bounds) 

    sample_acc_list = []
    credal_prob_matrix = []
    likelyhoods = []
    pram_smaple_list = []

    x_list = []
    y_list = []

    for params in x0:
        # print("random init params ", params)
        x_list.append(list(params.values()))
        cv_score, test_prob, likelyhood = sample_loss(d_param, params, x_train, y_train, x_test, y_test, seed)
        y_list.append(cv_score)

        sample_acc_list.append(cv_score)
        credal_prob_matrix.append(test_prob)
        likelyhoods.append(likelyhood)
        pram_smaple_list.append(params)


    xp = np.array(x_list)
    yp = np.array(y_list)

    # Create the GP
    if gp_params is not None:
        model = gp.GaussianProcessRegressor(**gp_params)
    else:
        kernel = gp.kernels.Matern()
        model = gp.GaussianProcessRegressor(kernel=kernel,
                                            alpha=alpha,
                                            n_restarts_optimizer=10,
                                            normalize_y=True)

    for n in range(n_iters):
        # print("------------------------------------xp")
        # print(xp)
        # print("------------------------------------yp")
        # print(yp)
        # print("------------------------------------")

        model.fit(xp, yp)

        # Sample next hyperparameter
        next_sample = sample_next_hyperparameter(expected_improvement, model, yp, greater_is_better=True, bounds=array_bounds, n_restarts=100)

        # convert next_sample to dict and also integer values
        next_sample_dict = x0[0]
        for pram, sample_value in zip(next_sample_dict, next_sample):
            next_sample_dict[pram] = int(sample_value)

        # Duplicates will break the GP. In case of a duplicate, we will randomly sample a next query point.
        if np.any(np.abs(next_sample - xp) <= epsilon):
            # print("Yes this is an exception!!!!!")
            next_sample_dict = random_pram_sample(bounds) # np.random.uniform(bounds[:, 0], bounds[:, 1], bounds.shape[0])
            next_sample = list(next_sample_dict.values())



        # Sample loss for new set of parameters
        cv_score, test_prob, likelyhood = sample_loss(d_param, next_sample_dict, x_train, y_train, x_test, y_test, seed)

        sample_acc_list.append(cv_score)
        credal_prob_matrix.append(test_prob)
        likelyhoods.append(likelyhood)
        pram_smaple_list.append(next_sample)

        # Update lists
        # print(">>> a ", next_sample)
        x_list.append(next_sample)
        y_list.append(cv_score)

        # Update xp and yp
        xp = np.array(x_list)
        yp = np.array(y_list)

    return sample_acc_list, credal_prob_matrix, likelyhoods, pram_smaple_list
