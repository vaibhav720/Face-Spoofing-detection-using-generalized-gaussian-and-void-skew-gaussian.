from typing import List, Tuple, Dict

import cv2
import matplotlib.pyplot as plt
import numpy as np
import scipy
from lmfit.models import SkewedVoigtModel

from sggd import sggd_cost, sggd, compute_sggd_variance


# Explain why won't you use the ParabolicModel
# 1. Mention required preprocessing step and introduction of a new hyper parameter
# 2. Same number of parameters as Gaussian which is three
# (https://www.radfordmathematics.com/functions/quadratic-functions-parabola/vertex-form/vertex-form-finding-equation-parabola.html)
# Note, the GaussianModel has magnitude, mean and sigma parameters

# https://jkillingsworth.com/2022/07/07/generalized-normal-distributions/
def create_gaussian_kernel(nb_g_sigma: float,
                           window_size: List[int]) -> np.ndarray:
    assert len(window_size) == 2
    assert window_size[0] == window_size[1]
    assert window_size[0] % 2 == 1

    g_sigma = window_size[0] / (2 * nb_g_sigma)

    gaussian_kernel = np.zeros(window_size, dtype=np.float32)

    half_window_size = (window_size[0] // 2)
    start = -half_window_size
    end = half_window_size

    for i in range(start, end + 1):
        for j in range(start, end + 1):
            x = j * g_sigma
            y = i * g_sigma
            scaler = 1 / (2 * np.pi * (g_sigma ** 2))
            dist_squared = x ** 2 + y ** 2
            gaussian_kernel[i + half_window_size, j + half_window_size] = scaler * np.exp(
                - (dist_squared / (2 * (g_sigma ** 2))))

    # normalize for unit volume
    gaussian_kernel = gaussian_kernel / gaussian_kernel.sum()

    return gaussian_kernel


def calculate_mscn_coefficients(face_gray_img: np.ndarray,
                                g_sigma_frac: float,
                                window_size: List[int]) -> np.ndarray:
    gaussian_kernel = create_gaussian_kernel(g_sigma_frac, window_size)

    pad_len = window_size[0] // 2
    face_gray_img_padded = np.pad(face_gray_img, pad_len, mode='edge')

    img_height, img_width = face_gray_img_padded.shape[:2]
    window_height, window_width = gaussian_kernel.shape[:2]

    half_window_height, half_window_width = window_height // 2, window_width // 2

    rescaled_img_height, rescaled_img_width = img_height - window_height + 1, img_width - window_width + 1

    mscn_coefficients = np.zeros((rescaled_img_height, rescaled_img_width), dtype=np.float32)

    for i in range(rescaled_img_height):
        for j in range(rescaled_img_width):
            img_i = i + half_window_height
            img_j = j + half_window_width

            img_window = face_gray_img_padded[i: i + window_height, j: j + window_width]
            img_window_mean = np.dot(gaussian_kernel, img_window).sum()
            img_window_sigma = np.sqrt(np.dot(gaussian_kernel,
                                              (img_window - img_window_mean) ** 2).sum())
            mscn_coefficients[i, j] = ((face_gray_img_padded[img_i, img_j] - img_window_mean) /
                                       (img_window_sigma + 1))

    return mscn_coefficients


def get_neighbor_directions(nb_neighbors=8) -> List[Tuple[int, int]]:
    if nb_neighbors == 8:
        directions = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0:
                    continue
                directions.append((i, j))
        return directions
    else:
        raise ValueError('Invalid value for nb_neighbors')


def calculate_adjacent_mscn(mscn_coefficients: np.ndarray,
                            neighbor_directions: List[Tuple[int, int]]) -> np.ndarray:
    mscn_coefficients_padded = np.pad(mscn_coefficients, 1, mode='edge')
    nb_neighbors = len(neighbor_directions)

    h, w = mscn_coefficients.shape[:2]
    mscn_coefficients_adj = np.zeros((h, w, nb_neighbors), dtype=np.float32)

    for i in range(h):
        for j in range(w):
            padded_i = i + 1
            padded_j = j + 1

            for idx, (i_inc, j_inc) in enumerate(neighbor_directions):
                mscn_coefficients_adj[i, j, idx] = mscn_coefficients_padded[padded_i, padded_j] * \
                                                   mscn_coefficients_padded[padded_i + i_inc, padded_j + j_inc]

    return mscn_coefficients_adj


def get_prewitt_y() -> np.ndarray:
    scaler: float = 1 / 3
    return np.array([[scaler, scaler, scaler], [0, 0, 0], [-scaler, -scaler, -scaler]])


def get_prewitt_x() -> np.ndarray:
    return get_prewitt_y().T


def get_grad_magnitude(img_gray: np.ndarray) -> np.ndarray:
    edge_x_filter = np.fliplr(np.flipud(get_prewitt_x()))
    edge_y_filter = np.fliplr(np.flipud(get_prewitt_y()))

    img_grad_x = cv2.filter2D(img_gray, -1, edge_x_filter)
    img_grad_y = cv2.filter2D(img_gray, -1, edge_y_filter)

    return np.sqrt(img_grad_x ** 2 + img_grad_y ** 2)


def calculate_epsd(face_gray_img: np.ndarray,
                   g_sigma: float = .6,
                   p: float = 5,
                   c: float = 1e-6) -> float:
    img_gaussian_blurred = cv2.GaussianBlur(face_gray_img,
                                            (0, 0),
                                            sigmaX=g_sigma, sigmaY=g_sigma,
                                            borderType=cv2.BORDER_REPLICATE)

    img_mag = get_grad_magnitude(face_gray_img)
    img_gaussian_blurred_mag = get_grad_magnitude(img_gaussian_blurred)

    gms = ((2 * img_mag * img_gaussian_blurred_mag) + c) / ((img_mag ** 2 + img_gaussian_blurred_mag ** 2) + c)
    th = np.percentile(gms.flatten(), 100 - p)
    gms[gms < th] = 0

    gms = gms.flatten()
    m_eps = gms.mean()
    epsd = np.sqrt(((gms - m_eps) ** 2).mean())

    return epsd


def compute_discrete_probability_distribution(data: np.ndarray, hist_nb_bins: int) -> Tuple[np.ndarray, np.ndarray]:
    counts_perc, bin_ranges = np.histogram(data, bins=hist_nb_bins, density=True)
    bin_sizes = np.diff(bin_ranges)

    x = bin_ranges[:-1]
    x += bin_sizes / 2

    y = counts_perc * bin_sizes

    return x, y


# # fn = 'temp/bf09088bcdda4ed6853c6db0c24a9372_0_face.jpg'
# fn = 'temp/a6bf05c8ab214480a80efcebdc9ab449_1_face.jpg'
# # # epsd = calculate_epsd(cv2.imread(fn, cv2.IMREAD_GRAYSCALE), 3)
# #
# mscn_coefficients = calculate_mscn_coefficients(cv2.imread(fn, cv2.IMREAD_GRAYSCALE),
#                                                 3,
#                                                 [7, 7])
#
#
# # np.save('mscn.npy', mscn_coefficients)
# mscn_coefficients = np.load('mscn.npy')
#
# adj_mscn = calculate_adjacent_mscn(mscn_coefficients, get_neighbor_directions(8))
# # np.save('adj_mscn.npy', adj_mscn)
#

def estimate_skewed_voigt_params(data: np.ndarray, hist_nb_bins: int = 50) -> Dict[str, float]:
    x, y = compute_discrete_probability_distribution(data, hist_nb_bins)

    # plt.scatter(x, y)

    model = SkewedVoigtModel()
    params_init = model.guess(y, x)
    res = model.fit(y, params_init, x=x)

    # plt.plot(x, res.best_fit)
    # plt.show()

    fit_summary = res.summary()
    return fit_summary['best_values']


def estimate_sggd_params(data: np.ndarray) -> Dict[str, float]:
    beta_init = 1
    mean_init = data.mean()
    alpha_init = np.sqrt(((data - mean_init) ** 2).sum() * (2 / len(data)))

    params_init = np.array([beta_init, mean_init, alpha_init])

    res = scipy.optimize.minimize(sggd_cost, params_init, method='nelder-mead', args=(data,))

    param_estimates = res.get('x')

    x, y = compute_discrete_probability_distribution(data, 100)

    # plt.scatter(x, y)

    sggd_y = sggd(param_estimates, x)
    sggd_y = sggd_y / sggd_y.sum()

    # plt.plot(x, sggd_y)
    # plt.show()


    return {
        'beta': param_estimates[0],
        'mean': param_estimates[1],
        'alpha': param_estimates[2],
    }


def extract_mscn_features_from_face_img(face_img: np.ndarray) -> List[float]:
    features = []
    mscn = calculate_mscn_coefficients(face_img,
                                       3,
                                       [7, 7])

    sggd_param_estimates = estimate_sggd_params(mscn.flatten())
    features.append(sggd_param_estimates['mean'])
    features.append(compute_sggd_variance(beta=sggd_param_estimates['beta'], alpha=sggd_param_estimates['alpha']))

    adj_mscn = calculate_adjacent_mscn(mscn, get_neighbor_directions(8))
    for i in range(adj_mscn.shape[2]):
        skewed_voigt_param_estimates = estimate_skewed_voigt_params(adj_mscn[:, :, i].flatten(), 100)
        features.append(skewed_voigt_param_estimates['skew'])

    return features


def extract_features_from_img(img: np.ndarray, face_xywh: List[int]) -> List[float]:
    x1, y1 = face_xywh[0], face_xywh[1]
    x2, y2 = x1 + face_xywh[2], y1 + face_xywh[3]
    face_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[y1: y2, x1: x2]
    features = extract_mscn_features_from_face_img(face_img)

    f_h, f_w = face_img.shape[:2]
    face_img_filtered = cv2.resize(face_img, (f_w // 2, f_h // 2))
    face_img_filtered = cv2.GaussianBlur(face_img_filtered, (7, 7), 0)
    features.extend(extract_mscn_features_from_face_img(face_img_filtered))

    features.append(calculate_epsd(face_img, 3))

    return features


if __name__ == '__main__':
    # mscn_coefficients = np.load('mscn.npy').flatten()
    fn = '../random_samples/e7ee29be861747ca915f358758fb309e_0_face.jpg'
    fn = '../random_samples/e602082b89c44b2bb2774ed16b8d64f9_1_face.jpg'
    face_img = cv2.cvtColor(cv2.imread(fn), cv2.COLOR_BGR2GRAY)
    calculate_epsd(face_img)
    # f = extract_features_from_img(cv2.imread(fn), [200, 200, 300, 300])
    # print(f)
    # mscn_coefficients = calculate_mscn_coefficients(cv2.imread(fn, cv2.IMREAD_GRAYSCALE),
    #                                                 3,
    #                                                 [7, 7])
    # print(estimate_sggd_params(mscn_coefficients))
