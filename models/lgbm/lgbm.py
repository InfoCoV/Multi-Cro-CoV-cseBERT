"""LGBM model definition and experiment setup"""

import logging
import warnings

import lightgbm as lgb
import numpy as np
from sklearn.metrics import classification_report, f1_score
from omegaconf import OmegaConf
import optuna
from optuna.samplers import TPESampler

from optuna_utils import OptunaExperiments, run_experiments

LOGGER = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


def lgb_f1_score(y_true, y_pred):
    y_pred = np.round(y_pred)
    return 'f1', f1_score(y_true, y_pred), True


config = OmegaConf.create(
    {"clf_params": {
        "boosting_type": "gbdt", "subsample_for_bin": 200000,
        "objective": "binary", "class_weight": "balanced",
        "random_state": 1, "n_jobs": -1, "subsample_freq": 1,
        "device_type": "cpu", "verbosity": 1},
     "num_leaves_min":  100,
     "num_leaves_max": 1100,
     "num_leaves_step": 200,
     "max_depth_min": 9,
     "max_depth_max": 21,
     "max_depth_step": 3,
     "min_child_samples_min": 5,
     "min_child_samples_max": 20,
     "min_child_samples_step": 5,
     "n_estimators_min": 1000,
     "n_estimators_max": 2000,
     "n_estimators_step": 1000,
     "learning_rates": [0.01, 0.001],
     "reg_alpha_min": 0,
     "reg_alpha_max": 20,
     "reg_alpha_step": 2,
     "reg_lambda_min": 0,
     "reg_lambda_max": 20,
     "reg_lambda_step": 2,
     "min_split_gain_min": 0,
     "min_split_gain_max": 10,
     "subsample_min": 0.2,
     "subsample_max": 1.0,
     "subsample_step": 0.1,
     "colsample_bytree_min": 0.1,
     "colsample_bytree_max": 1.0,
     "colsample_bytree_step": 0.1,
     "fit_verbose": True}
)


class Experiments(OptunaExperiments):

    def create_study(self):
        sampler = TPESampler(seed=self.config.study_seed)
        study = optuna.create_study(sampler=sampler, direction="maximize")
        for trial_dict in self.config.default_trials:
            study.enqueue_trial(trial_dict)
        return study

    def optimize(self):
        self.study.optimize(self.objective, n_trials=self.config.n_trials)
        self.store_study()

    def objective(self, trial):
        num_leaves = trial.suggest_int(
            "num_leaves", self.config.num_leaves_min,
            self.config.num_leaves_max, self.config.num_leaves_step)
        max_depth = trial.suggest_int(
            "max_depth", self.config.max_depth_min, self.config.max_depth_max,
            self.config.max_depth_step
        )
        min_child_samples = trial.suggest_int(
            "min_child_samples", self.config.min_child_samples_min,
            self.config.min_child_samples_max,
            self.config.min_child_samples_step
        )
        n_estimators = trial.suggest_int(
            "n_estimators", self.config.n_estimators_min,
            self.config.n_estimators_max,
            self.config.n_estimators_step
        )
        learning_rate = trial.suggest_categorical(
            "learning_rate",
            self.config.learning_rates
        )
        reg_alpha = trial.suggest_int(
            "reg_alpha", self.config.reg_alpha_min,
            self.config.reg_alpha_max, self.config.reg_alpha_step
        )
        reg_lambda = trial.suggest_int(
            "reg_lambda", self.config.reg_lambda_min,
            self.config.reg_lambda_max, self.config.reg_lambda_step
        )
        min_split_gain = trial.suggest_int(
            "min_split_gain", self.config.min_split_gain_min,
            self.config.min_split_gain_max
        )
        subsample = trial.suggest_float(
            "subsample", self.config.subsample_min,
            self.config.subsample_max,
            step=self.config.subsample_step)
        colsample_bytree = trial.suggest_float(
            "colsample_bytree", self.config.colsample_bytree_min,
            self.config.colsample_bytree_max,
            step=self.config.colsample_bytree_step
        )

        params = {
            "num_leaves": num_leaves,
            "max_depth": max_depth,
            "min_child_samples": min_child_samples,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "min_split_gain": min_split_gain,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree
        }
        LOGGER.info(params)

        pruning = optuna.integration.LightGBMPruningCallback(
            trial, "f1", valid_name="val")
        clf = lgb.LGBMClassifier(
            num_leaves=num_leaves,
            max_depth=max_depth,
            min_child_samples=min_child_samples,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_split_gain=min_split_gain,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            **self.config.clf_params)

        clf.fit(
            self.train_data,
            self.train_labels,
            eval_set=[(self.train_data, self.train_labels),
                      (self.val_data, self.val_labels)],
            eval_metric=lgb_f1_score,
            eval_names=["train", "val"],
            callbacks=[pruning],
            early_stopping_rounds=n_estimators // 10,
            verbose=self.config.fit_verbose
        )
        val_predictions = clf.predict(self.val_data)

        out = classification_report(
            self.val_labels.values, val_predictions,
            digits=3, output_dict=True)
        LOGGER.info(out)
        f1 = out["macro avg"]["f1-score"]

        if self.best_score is None or f1 > self.best_score:
            self.best_score = f1
            self.store_results(clf, out)
            self.store_study()

        return f1


def run(config):
    run_experiments(
        config=config,
        experiments_class=Experiments)
