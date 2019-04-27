"""
Cross Validation
----------------

Cross-validation support.

We assume that instances (i) are uniquely identified by an instance ID and (ii) they have id and score properties.
We access them using the Instances class.

:Author: Faegheh Hasibi
"""

import json
import numpy as np
import pandas as pd
from sklearn import metrics, model_selection
from sklearn.utils import shuffle

from ovlpred.ml.train_test import TrainTest


class CrossValidation(object):
    """
    Class attributes:
        fold: dict of folds (1..k) with a dict
              {"train": [list of instance_ids]}, {"test": [list of instance_ids]}

    """

    def __init__(self, k, x, y, callback_train, callback_normalize=None):
        """
        :param k: number of folds
        :param x: training instances
        :param y: target values
        :param callback_train: Callback function for training model
        """
        self.__k = k
        self.__x = x
        self.__y = y
        self.__folds = None
        self.__callback_train = callback_train
        self.__callback_normalize = callback_normalize

    @staticmethod
    def create_folds(data, k, group_by=None):
        """Creates folds for the data set.

        :param data: data frame
        :param k: number of folds
        :param group_by: name of the feature, used for splitting the data into train/test set.
                         Instances with the same value for that features are not spread across different folds.
        """
        # if group_by is not specified, then grouping will be based on the X indices (i.e., no grouping)
        groups = data[group_by] if group_by else data.index

        # determines the number of folds; one-leave-out if k=-1
        num_folds = data.shape[0] if k == -1 else k

        # shuffles X indices and the groups
        x_shuffled, groups_shuffled = shuffle(data.index, list(groups))
        # creates the folds
        i = 0
        folds = {}
        kf = model_selection.GroupKFold(num_folds)
        for train_index, test_index in kf.split(x_shuffled, groups=groups_shuffled):
            print("Generating fold " + str(i + 1) + "/" + str(num_folds))
            folds[str(i)] = {"train": sorted([int(x_shuffled[x]) for x in train_index]),
                             "test": sorted([int(x_shuffled[x]) for x in test_index])}
            i += 1
        return folds

    def get_folds(self, splits_file, data=None, group_by=None, create_splits=False):
        """Loads folds from file or generates them if specified.

        :param data: Train data (needed when creating startified folds)
        :param splits_file: name of the splits file
        :param group_by: number of folds
        :param create_splits: if true, generates new folds
        """
        # creates folds and saves them
        if create_splits:
            print("Creating splits ...")
            train_data = data if data is not None else self.__x
            self.__folds = self.create_folds(train_data, self.__k, group_by)
            # with open(splits_file, "w") as outfile:
            json.dump(self.__folds, open(splits_file, "w"), indent=4)
        # loads already created folds
        else:
            json_data = open(splits_file)
            self.__folds = json.load(json_data)

    def run(self, imp=False):
        """Runs cross-validation."""
        # Todo: Add feature normalization
        # if folds haven't been initialized/created before (w/ get_folds or create_folds)
        # then they'll be created using the default grouping (i.e., based on instance_id)
        if self.__folds is None:
            self.create_folds()

        # this holds the estimated target values
        y_pred = pd.DataFrame(np.zeros(self.__y.shape))
        mse, r2 = [], []
        importance = pd.DataFrame(columns=self.__x.columns)
        for i, fold in sorted(self.__folds.items()):
            print("=======================================")
            print("Cross validation for fold " + str(i) + " ...")
            train_ind, test_ind = fold["train"], fold["test"]
            x_train, x_test = self.__x.iloc[train_ind, :], self.__x.iloc[test_ind, :]
            # normalization
            if self.__callback_normalize is not None:
                x_train, x_test, _ = self.__callback_normalize(x_train, x_test)

            model = self.__callback_train(x_train, self.__y.iloc[train_ind])
            y_pred.iloc[test_ind] = model.predict(x_test).reshape(len(test_ind), self.__y.shape[1])
            if imp:
                imp_df = pd.DataFrame([list(model.feature_importances_)], columns=self.__x.columns)
                importance = importance.append(imp_df, ignore_index=True)
            mse_fold, r2_fold = TrainTest.get_performance(self.__y.iloc[test_ind], y_pred.iloc[test_ind])
            mse.append(mse_fold)
            r2.append(r2_fold)
            print("R2: ", np.round(r2_fold, 4))
        print("=======================================")
        r2, mse = np.vstack(r2), np.vstack(mse)
        print("R2-mean: ", np.round(np.mean(r2, axis=0), 4), "\nR2-std: ", np.round(np.std(r2, axis=0), 4))
        print("MSE-mean: ", np.round(np.mean(mse, axis=0), 4), "\nMSE-std: ", np.round(np.std(mse, axis=0), 4))
        return {"mse": mse, "r2": r2, "y_pred": y_pred, "y": self.__y, "imp": importance.mean().transpose().to_dict()}
