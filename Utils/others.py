import numpy as np
import matplotlib.pyplot as plt

def get_average(input):
    for i in range(len(input)):
        if i == 0:
            w_arr = abs(input[i])
        else:
            w_arr = np.concatenate((w_arr, abs(input[i]) ))
    return w_arr.mean()

def get_max(input):
    max_arr = [abs(input[i]).max() for i in range(len(input))]
    return max(max_arr)