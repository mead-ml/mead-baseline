import json
import tensorflow as tf
import numpy as np

def revlut(lut):
    return {v: k for k, v in lut.items()}



# Fill out the y matrix with 0s for other labels
def fill_y(nc, yidx):
    xidx = np.arange(0, yidx.shape[0], 1)
    dense = np.zeros((yidx.shape[0], nc))
    dense[xidx, yidx] = 1
    return dense

