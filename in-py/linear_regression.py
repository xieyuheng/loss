import numpy as np
import pandas as pd

from learner import Learner

def hypo(parameters, data_row):
    return data_row.dot(parameters)

def loss(parameters, data_frame, targets):
    diff = data_frame.dot(parameters) - targets
    return (diff * diff / 2).mean()

def row_gradient(parameters, data_row, target):
    return (data_row.dot(parameters) - target) * data_row
