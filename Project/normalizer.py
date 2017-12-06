import pandas as pd
import numpy as np


def generic_normalize(x, define_max=0):
    norm_min = x.min()
    if define_max == 0:
        max = x.max()
    else:
        max = define_max
    diff = max - norm_min

    return ((x - norm_min) / diff) - 0.5, norm_min, diff


def generic_denormalize(x, denorm_min, diff):
    return ((x + 0.5) * diff) + denorm_min


