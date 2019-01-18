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
                self.row_gradient(
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
