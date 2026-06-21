from sklearn.metrics.pairwise import rbf_kernel, linear_kernel, polynomial_kernel
import numpy as np
from utils import StandardizeKernel

def get_kernel(X, rbf_gamma_list=None, poly_params=None, use_linear=False):

    if rbf_gamma_list is None:  rbf_gamma_list = []

    if poly_params is None: poly_params = []

    if len(rbf_gamma_list) == 0 and len(poly_params) == 0 and not use_linear:
        raise ValueError("At least one kernel must be selected")
    
    K = []

    for gamma in rbf_gamma_list:
        K_rbf = rbf_kernel(X, gamma=gamma)
        K_rbf_std = StandardizeKernel(K_rbf, center=True, scale=True)
        K.append(K_rbf_std)

    for coef0, degree in poly_params:
        K_poly = polynomial_kernel(X, degree=degree, coef0=coef0, gamma=1)
        K_poly_std = StandardizeKernel(K_poly, center=True, scale=True)
        K.append(K_poly_std)

    if use_linear:
        K_linear = linear_kernel(X)
        K_linear_std = StandardizeKernel(K_linear, center=True, scale=True)
        K.append(K_linear_std)
   
    K_tensor = np.stack(K, axis=2)
    return K_tensor
