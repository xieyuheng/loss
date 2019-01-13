import numpy as np
import pandas as pd

class Learner:
    def __init__(self, df, targets,
                 hypo, loss, row_gradient):

        assert (type(df) == pd.DataFrame)
        assert (type(targets) == pd.Series)

        self.df = df
        self.df.loc["bias"] = 1

        self.targets = targets

        # -- parameters, data_row -> target
        self.hypo = hypo

        # -- parameters, data_frame, targets -> float
        self.loss = loss

        # -- parameters, data_row, target -> gradient
        self.row_gradient = row_gradient

    # -- parameters, data_frame, targets, learning_rate -> parameters
    def step(self, parameters, data_frame, targets, learning_rate):
        gradient_list = []
        for i in range(targets.size):
            gradient_list.append(
                row_gradient(
                    parameters,
                    data_frame.loc[i],
                    targets[i]))
        gradient = pd.DataFrame(gradient_list).mean()
        return parameters - learning_rate * gradient

    def fit(self, steps, learning_rate):
        parameters = pd.Series(
            np.zeros(self.df.columns.size),
            index=self.df.columns)
        for _ in range(steps):
            parameters = self.step(
                parameters, self.df, self.targets, learning_rate)
        return parameters

df = pd.DataFrame({
    "area": [2104, 1416, 1534, 852, 124],
    "#bedrooms": [5, 3, 3, 2, 1],
    "#floors": [1, 2, 2, 1, 1],
    "age": [45, 40, 30, 36, 20],
})

targets = pd.Series([460, 232, 315, 178, 130])

def hypo(parameters, data_row):
    return data_row.dot(parameters)

def loss(parameters, data_frame, targets):
    diff = data_frame.dot(parameters) - targets
    return (diff * diff / 2).mean()

def row_gradient(parameters, data_row, target):
    return (data_row.dot(parameters) - target) * data_row

learner = Learner(df, targets, hypo, loss, row_gradient)

learner.fit(1000, 0.0000001)
