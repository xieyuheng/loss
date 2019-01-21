import numpy as np
import pandas as pd

from learner import Learner

import linear_regression

def test ():
    df = pd.DataFrame ({
        "area": [2104, 1416, 1534, 852, 124],
        "#bedrooms": [5, 3, 3, 2, 1],
        "#floors": [1, 2, 2, 1, 1],
        "age": [45, 40, 30, 36, 20],
    })

    targets = pd.Series ([460, 232, 315, 178, 130])

    learner = Learner (
        df, targets,
        linear_regression.hypo,
        linear_regression.loss,
        linear_regression.row_gradient)

    print (learner.fit (100, 0.0000001))
