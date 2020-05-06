import numpy as np
import torch

## get_relative_l2_error
def get_relative_l2_error(data_inf, data_ref):
    ## data_inf and data_ref have the same size (num_t, num_x, num_y)
    [num_t, num_x, num_y] = data_inf.shape
    error = np.zeros(num_t)
    var = data_ref.var()
    for t in range(0, num_t):
        err = ((data_inf[t] - data_ref[t])**2).mean()
        error[t] = err/var
    return error

## get_relative_l2_error
def get_l2_error(data_inf, data_ref):
    ## data_inf and data_ref have the same size (num_t, num_x, num_y)
    [num_t, num_x, num_y] = data_inf.shape
    error = np.zeros(num_t)
    for t in range(0, num_t):
        error[t] = ((data_inf[t] - data_ref[t])**2).mean()
    return error

def to_numpy(input):
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or '\
            'np.ndarray, but got {}'.format(type(input)))