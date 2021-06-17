import os, pickle, string
import pandas as pd
import numpy as np
import rampwf as rw
from sklearn.utils import shuffle
from rampwf.score_types.base import BaseScoreType


problem_title = "Nuclear inventory of a nuclear reactor core in operation"

_target_names = ["Y_" + j for j in list(string.ascii_uppercase)]

Predictions = rw.prediction_types.make_regression(label_names=_target_names)
workflow = rw.workflows.Regressor()


class inventoryError_MSE(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float("inf")

    def __init__(self, name="this error", precision=3):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        mse = (np.square(y_true - y_pred).sum(axis=1)).mean()
        return mse


# ----


class inventoryError_MAE(BaseScoreType):
    is_lower_the_better = True
    minimum = 0.0
    maximum = float("inf")

    def __init__(self, name="this error", precision=3):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        mape = (np.fabs(y_true - y_pred)).mean().sum()
        return mape


# ----


score_types = [
    inventoryError_MSE(name="inventoryError_MSE"),
    inventoryError_MAE(name="inventoryError_MAE"),
]


def get_train_data(path="."):

    # load pre-prepared dataset aggregating all of the different input data
    # ( for the training dataset, these are composed of 920 different simulation of an operating reactor )
    # train_dataset = pickle.load( open( "./data/train_data_python3.pickle", "rb") )
    train_dataset = pickle.load(
        open(os.path.join(path, "data", "train_data_python3.pickle"), "rb")
    )

    train_dataset = train_dataset / train_dataset.max()  # normalize data

    # Isotopes are named from A to Z
    alphabet = list(string.ascii_uppercase)

    # At T=0 only isotopes from A to H are != 0. Those are the input parameters
    # The input parameter space is composed of those initial compositions
    input_params = alphabet[:8] + ["p%d" % (i) for i in range(1, 6)]

    train_data = train_dataset[alphabet].add_prefix("Y_")
    train_data["times"] = train_dataset["times"]

    temp = pd.DataFrame(
        np.repeat(train_dataset.loc[0][input_params].values, 81, axis=0),
        columns=input_params,
    ).reset_index(drop=True)
    train_data = pd.concat([temp, train_data.reset_index(drop=True)], axis=1)

    train_data = shuffle(train_data, random_state=57)

    return (
        train_data[input_params + ["times"]].to_numpy(),
        train_data[["Y_" + j for j in alphabet]].to_numpy(),
    )


def get_test_data(path="."):

    # load pre-prepared dataset aggregating all of the different input data
    # ( for the testing dataset, these are composed of 200 different simulation of an operating reactor )
    test_dataset = pickle.load(
        open(os.path.join(path, "data", "test_data_python3.pickle"), "rb")
    )

    test_dataset = test_dataset / test_dataset.max()  # normalize data

    # Isotopes are named from A to Z
    alphabet = list(string.ascii_uppercase)

    # At T=0 only isotopes from A to H are != 0. Those are the input parameters
    # The input parameter space is composed of those initial compositions
    input_params = alphabet[:8] + ["p%d" % (i) for i in range(1, 6)]

    test_data = test_dataset[alphabet].add_prefix("Y_")
    test_data["times"] = test_dataset["times"]

    temp = pd.DataFrame(
        np.repeat(test_dataset.loc[0][input_params].values, 81, axis=0),
        columns=input_params,
    ).reset_index(drop=True)
    test_data = pd.concat([temp, test_data.reset_index(drop=True)], axis=1)

    test_data = shuffle(test_data, random_state=57)

    return (
        test_data[input_params + ["times"]].to_numpy(),
        test_data[["Y_" + j for j in alphabet]].to_numpy(),
    )


def get_cv(X, y):
    return [
        (range(0, X.shape[0] - 200), range(X.shape[0] - 200, X.shape[0])),
    ]
