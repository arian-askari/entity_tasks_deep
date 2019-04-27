"""
Machine leaning
===============

Core functionality for general-purpose machine learning.


Usage
-----

::

  python -m nordlys.core.ml.ml <config_file>


Config parameters
------------------

- **train_data**: train set csv file,
- **features**: list of features to be considered; if not specified, all features will be considered
- **target**: label of target value
- **held_out**: list of features that are not considered for training
- **group_by**: list of features, which are used for building the splits. If not specified, iid sampling is performed.
- **Normalize**: if true, normalizes the feature set; default False
- **train_test**:
   - test_frac: fraction of data to be used for testing; default 0.2
   - val-frac: fraction of data to be used for testing; default 0
   - splits_file: Json file to save train-test-validation splits. holds the Id of instances
   - model_file: file to save/load the trained model
   - load_model: if true, applies the given model and does not perform training
   - create_splits: if True, creates the splits. Otherwise loads the splits from "splits_file" parameter.
- **cross_validation**:
   - k: number of folds (default: 5); use -1 for leave-one-out
   - splits_file: JSON file with splits (instance_ids); holds the ID of instances for each fold
   - create_splits: if True, creates the CV splits. Otherwise loads the splits from "split_file" parameter.
- **learning_curve**:
   - k: number of folds (default: 5)
   - metric: name of metric to report on; default: r2
   - ylim: range of y-axis in the plot; should be a list with two values; e.g., [0,1]
   - plot_file: name of the plot file
- **model**: ML model, currently supported values: rf, gbrt
- **category**: [regression | classification], default: "regression"
- **parameters**: dict with parameters of the given ML model
   - If GBRT:
      - learning_rate: default: 0.1
      - tree: number of trees, default: 1000
      - depth: max depth of trees, default: 10% of number of features
   - If RF:
      - tree: number of trees, default: 1000
      - maxfeat: max features of trees, default: 10% of number of features
   - If NN:
      - layers: number of nodes in each layer; default [10]
      - lambda: Regularization factor; default: 0
      - optimizer: name of optimizer; default: adam
      - batch_size: size of batch; default: 512
      - activation = name of activation function; default: "relu"
- **feature_imp_file**: Feature importance is saved to this file
- **model_file**: model is saved to this file
- **output_file**: name of output file; holds train-test score for learning curve,
                    and model predictions for train-test or cross validation

Example config
---------------

.. code:: python

{
    "model": "gbrt",
    "category": "regression",
    "parameters":{
        "learning_rate": 0.1,
        "tree": 10,
        "depth": 5
    },
    "cross_validation":{
        "create_splits": true,
        "splits_file": "path/to/splits.json",
        "k": 5
    },
    "train_data": "path/to/train.csv",
    "target": "name_of_feature",
    "held_out": ["f1", "f2", "f3"],
    "group_by": "q_id",
    "output_file": "path/to/output.json"
}

------------------------

:Authors: Faegheh Hasibi
"""
import argparse
import pandas as pd
from sys import exit
import pickle
import json
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.preprocessing import StandardScaler
from nordlys.core.ml.cross_validation import CrossValidation
from nordlys.core.ml.feedforward import Feedforward
from nordlys.core.ml.plotting import LearningCurve
from nordlys.core.ml.train_test import TrainTest
from nordlys.core.utils.file_utils import FileUtils


class ML(object):
    def __init__(self, config):
        self.__check_config(config)
        self.__config = config
        self.features = None

    @staticmethod
    def __check_config(config):
        """Checks config parameters and set default values."""
        try:
            if "target" not in config:
                raise Exception("target is missing!")
            if type(config["target"]) != list:
                raise Exception("target should be a list!")
            if "cross_validation" in config:
                if "splits_file" not in config["cross_validation"]:
                    raise Exception("splits_file is missing!")
            if "train_test" in config:
                if "splits_file" not in config["train_test"]:
                    raise Exception("splits_file is missing!")
                if "load_model" in config["train_test"] and "model_file" not in config["train_test"]:
                    raise Exception("model_file is missing!")
        except Exception as e:
            print("Error in config file: ", e)
            exit(1)

    @staticmethod
    def normalize(x_train, x_test, x_val=None):
        """Performs mean-variance standardization."""
        print("Normalizing data ...")
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train = pd.DataFrame(scaler.transform(x_train), columns=x_train.columns)
        x_test = pd.DataFrame(scaler.transform(x_test), columns=x_test.columns)
        if x_val is not None:  x_val = pd.DataFrame(scaler.transform(x_val), columns=x_val.columns)
        return x_train, x_test, x_val

    def get_features(self, train_data):
        if self.features is not None:
            return self.features
        if "features" in self.__config:
            features = self.__config["features"]
        elif "held_out" in self.__config:
            heldout = self.__config["target"] + self.__config["held_out"]
            features = list(set(train_data.columns) - set(heldout))
        else:
            features = list(train_data.columns)
        self.features = features
        return self.features

    def gen_model(self, num_features=None):
        """ Reads parameters and generates a model to be trained.

        :param num_features: int, number of features
        :return untrained ranker/classifier
        """
        model_name, model = self.__config["model"].lower(), None
        if model_name == "lr":
            model = self.__gen_lr()
        elif model_name == "gbrt":
            model = self.__gen_gbrt(num_features)
        elif model_name == "rf":
            model = self.__gen_rf(num_features)
        elif (model_name == "mlp") or (model_name == "feedforward"):
            model = self.__gen_nn()
        return model

    def __gen_lr(self):
        """Generates a linear regression model."""
        print("Training using linear regression ...")
        model = LinearRegression()
        return model

    def __gen_gbrt(self, num_features):
        """Generates a GBRT regression/classification model."""
        learning_rate = self.__config["parameters"].get("learning_rate", 0.1)
        tree = self.__config["parameters"].get("tree", 1000)
        default_depth = round(num_features / 10.0) if num_features is not None else None
        depth = self.__config["parameters"].get("depth", default_depth)

        print("Number of trees: " + str(tree) + "\tDepth of trees: " + str(depth))
        if self.__config.get("category", "regression") == "regression":
            print("Training using GBRT regressor ...")
            model = GradientBoostingRegressor(n_estimators=tree, max_depth=depth, learning_rate=learning_rate)
        else:
            print("Training using GBRT classifier ...")
            model = GradientBoostingClassifier(n_estimators=tree, max_depth=depth, learning_rate=learning_rate)
        return model

    def __gen_rf(self, num_features):
        """Generates a GBRT regression/classification model."""
        tree = self.__config["parameters"].get("tree", 1000)
        default_maxfeat = round(num_features / 10.0) if num_features is not None else None
        max_feat = self.__config["parameters"].get("maxfeat", default_maxfeat)

        print("Number of trees: " + str(tree) + "\tMax features: " + str(max_feat))
        if self.__config.get("category", "regression") == "regression":
            print("Training using RF regressor ...")
            model = RandomForestRegressor(n_estimators=tree, max_features=max_feat)
        else:
            print("Training using RF classifier ...")
            model = RandomForestClassifier(n_estimators=tree, max_features=max_feat)
        return model

    def __gen_nn(self):
        """Generates a Feed forward Neural Net."""
        model_name = self.__config["model"].lower()
        category = self.__config.get("category", "regression")
        learning_rate = self.__config["parameters"].get("learning_rate", 0.1)
        layers = self.__config["parameters"].get("layers", [10])
        lambd = self.__config["parameters"].get("lambda", 0.)
        optimizer = self.__config["parameters"].get("optimizer", "adam")
        batch_size = self.__config["parameters"].get("batch_size", 512)
        activation = self.__config["parameters"].get("activation", "relu")
        epochs = self.__config["parameters"].get("epochs", 1000)
        if (model_name == "mlp") and (category == "regression"):
            print("Training using NN regressor ...")
            model = MLPRegressor(hidden_layer_sizes=tuple(layers), solver=optimizer, learning_rate_init=learning_rate,
                                 alpha=lambd, batch_size=batch_size, activation=activation,
                                 max_iter=epochs, learning_rate="constant", verbose=True)
        elif (model_name == "mlp") and (category == "classification"):
            print("Training using NN classifier ...")
            model = MLPClassifier(hidden_layer_sizes=tuple(layers), solver=optimizer, learning_rate_init=learning_rate,
                                  alpha=lambd, batch_size=batch_size,  activation=activation,
                                  max_iter=epochs, learning_rate="constant", verbose=True)
        elif model_name == "feedforward":
            layers = [len(self.features)] + layers + [len(self.__config["target"])]
            model = Feedforward(layers=layers, category=category, optimizer=optimizer, learning_rate=learning_rate,
                                activation=activation, lambd=lambd, batch_size=batch_size, epochs=epochs)
        return model

    def train(self, x, y, val_x=None, val_y=None):
        """Trains model on a given set of instances.

        :param x: training instances
        :param y: target values
        :param val_x: validation instances (may be used for NN training)
        :param val_y: validation target values
        :return: the learned model
        """
        n_instances, n_features = x.shape[0], x.shape[1]
        print("Number of instances:\t" + str(n_instances))
        print("Number of features:\t" + str(n_features))
        # training
        if self.__config.get("train_test", {}).get("load_model", False):
            pkl_file = open(self.__config["train_test"]["model_file"], "rb")
            model = pickle.load(pkl_file)
        else:
            model = self.gen_model(n_features)
            model.fit(x, y, val_x, val_y) if self.__config["model"] == "feedforward" else model.fit(x, y)
            if "model_file" in self.__config:
                print("Writing trained model to {} ...".format(self.__config["model_file"]))
                pickle.dump(model, open(self.__config["model_file"], "wb"))
        return model

    def analyse_features(self, importance):
        """ Ranks features based on their importance.
        Scikit uses Gini score to get feature importance.

        :param importance: data frame
        """
        sorted_importance = sorted(importance.items(), key=lambda imps: imps[1], reverse=True)
        feat_imp_str = ""
        for feat, importance in sorted_importance:
            feat_imp_str += feat + "\t" + str(importance) + "\n"
        open(self.__config["feature_imp_file"], "w").write(feat_imp_str)
        print("Feature importance file:", self.__config["feature_imp_file"])
        return sorted_importance

    def run(self):
        # loads data
        train_data = pd.read_csv(self.__config["train_data"], index_col=False, low_memory=False)
        features, target = self.get_features(train_data), self.__config["target"]
        print(features)
        x, y = train_data[features], train_data[target]
        group_by = self.__config.get("group_by", None)
        output_file = self.__config.get("output_file", None)
        feature_imp = "feature_imp_file" in self.__config
        normalize = self.normalize if self.__config.get("normalize", False) else None

        # Learning curve plot
        if "learning_curve" in self.__config:
            model = self.gen_model(x.shape[1])
            k = self.__config["learning_curve"]["k"]
            metric = self.__config["learning_curve"].get("metric", "r2")
            ylim = self.__config["learning_curve"].get("ylim", [0, 1])
            plot_file = self.__config["learning_curve"].get("plot_file", None)
            res = LearningCurve(k, train_data, features, target).plot(model, plot_file, group_by=group_by,
                                                                      ylim=(ylim[0], ylim[1]), metric=metric)
            if output_file: json.dump(res, open(output_file, "w"), indent=4)

        # Cross Validation
        elif "cross_validation" in self.__config:
            k = self.__config["cross_validation"].get("k", 5)
            split_file = self.__config["cross_validation"]["splits_file"]
            create_splits = self.__config["cross_validation"].get("create_splits", False)
            cv = CrossValidation(k, x, y, self.train, callback_normalize=normalize)
            cv.get_folds(split_file, data=train_data, group_by=group_by, create_splits=create_splits)
            res = cv.run(feature_imp)
            res["y_pred"].to_csv(output_file, index=False)

        # classic test-train split
        elif "train_test" in self.__config:
            splits_file = self.__config["train_test"]["splits_file"]
            create_splits = self.__config["train_test"].get("create_splits", False)
            test_frac = self.__config["train_test"].get("test_frac", 0.2)
            val_frac = self.__config["train_test"].get("val_frac", 0)
            # pred_test = self.__config["train_test"].get("pred_test", True)
            # pred_val = self.__config["train_test"].get("pred_val", True)
            tt = TrainTest(train_data, features, target, self.train, callback_normalize=normalize)
            tt.get_splits(splits_file, test_frac, val_frac, group_by=group_by, create_splits=create_splits)
            res = tt.run(feature_imp)
            res["test_pred"].to_csv(output_file, index=False)

        # writes feature importance if needed
        if "feature_imp_file" in self.__config:
            self.analyse_features(res["imp"])
        return res


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="config file", type=str)
    args = parser.parse_args()
    return args


def main(args):
    config = FileUtils.load_config(args.config)
    ml = ML(config)
    ml.run()


if __name__ == "__main__":
    main(arg_parser())
