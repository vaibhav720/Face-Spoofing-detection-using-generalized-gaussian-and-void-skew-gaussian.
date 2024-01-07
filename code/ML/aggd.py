from typing import Dict

import cv2
import numpy as np
import scipy
from matplotlib import pyplot as plt

from feature_extract_helpers import compute_discrete_probability_distribution, \
    calculate_mscn_coefficients, calculate_adjacent_mscn, get_neighbor_directions


def aggd(params: np.ndarray, x: np.ndarray):
    beta = params[0]
    mean = params[1]
    alpha1 = params[2]
    alpha2 = params[3]

    exponent = np.where(x < mean, (mean - x) / alpha1, (x - mean) / alpha2)
    scalar = beta / ((alpha1 + alpha2) * scipy.special.gamma(1 / beta))
    ggd = scalar * np.exp(-(exponent ** beta))
    return ggd


def aggd_cost(params: np.ndarray, data: np.ndarray):
    return -np.log(aggd(params, data)).sum()


def estimate_aggd_params(data: np.ndarray) -> Dict[str, float]:
    beta_init = 1
    mean_init = data.mean()
    alpha_init = np.sqrt(((data - mean_init) ** 2).sum() * (2 / len(data)))

    params_init = np.array([beta_init, mean_init, alpha_init, alpha_init])

    res = scipy.optimize.minimize(aggd_cost, params_init, method='nelder-mead', args=(data,))

    param_estimates = res.get('x')

    x, y = compute_discrete_probability_distribution(data, 100)

    plt.scatter(x, y)

    aggd_y = aggd(param_estimates, x)
    aggd_y = aggd_y / aggd_y.sum()

    plt.plot(x, aggd_y)
    plt.show()

    return {
        'beta': param_estimates[0],
        'mean': param_estimates[1],
        'alpha1': param_estimates[2],
        'alpha2': param_estimates[3],
    }


if __name__ == '__main__':
    # mscn_coefficients = np.load('mscn.npy').flatten()
    fn = '../random_samples/e7ee29be861747ca915f358758fb309e_0_face.jpg'
    fn = '../random_samples/e3034b190565452485cc03d9a371bf23_0_face.jpg'
    fn = '../random_samples/e602082b89c44b2bb2774ed16b8d64f9_1_face.jpg'
    face_img = cv2.cvtColor(cv2.imread(fn), cv2.COLOR_BGR2GRAY)

    mscn = calculate_mscn_coefficients(face_img,
                                       3,
                                       [7, 7])
    adj_mscn = calculate_adjacent_mscn(mscn, get_neighbor_directions(8))
    for i in range(adj_mscn.shape[2]):
        estimate_aggd_params(adj_mscn[:, :, i].flatten())

    # sggd_param_estimates = estimate_aggd_params(mscn.flatten())
