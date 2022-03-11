"""Random forest model definition and experiment setup"""

import logging


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from omegaconf import OmegaConf
import optuna
from optuna.samplers import TPESampler

from optuna_utils import OptunaExperiments, run_experiments


LOGGER = logging.getLogger(__name__)

# updated by train.py before running
config = OmegaConf.create(
    {"clf_params": {
        "random_state": 1, "verbose": True,
        "n_jobs": -1, "oob_score": True,
        "class_weight": "balanced"},
     "n_estimators_min": 100,
     "n_estimators_max": 200,
     "n_estimators_step": 50,
     "max_depth_min": 10,
     "max_depth_max": 25,
     "min_samples_split_min": 0.00001,
     "min_samples_split_max": 0.001,
     "min_samples_leaf_min": 0.00001,
     "min_samples_leaf_max": 0.001,
     "max_samples_min": 0.2,
     "max_samples_max": 0.8,
     "max_features_min": 0.1,
     "max_features_max": 1.0,
     "min_impurity_decrease_min": 1e-7,
     "min_impurity_decrease_max": 0.001}
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
        n_estimators = trial.suggest_int(
            "n_estimators",
            self.config.n_estimators_min,
            self.config.n_estimators_max,
            self.config.n_estimators_step)
        max_depth = trial.suggest_int(
            "max_depth",
            self.config.max_depth_min,
            self.config.max_depth_max
        )
        min_samples_split = trial.suggest_float(
            "min_samples_split",
            self.config.min_samples_split_min,
            self.config.min_samples_split_max,
            log=True
        )
        min_samples_leaf = trial.suggest_float(
            "min_samples_leaf",
            self.config.min_samples_leaf_min,
            self.config.min_samples_leaf_max,
            log=True
        )
        max_samples = trial.suggest_float(
            "max_samples",
            self.config.max_samples_min,
            self.config.max_samples_max
        )
        max_features = trial.suggest_float(
            "max_features",
            self.config.max_features_min,
            self.config.max_features_max
        )
        min_impurity_decrease = trial.suggest_float(
            "min_impurity_decrease",
            self.config.min_impurity_decrease_min,
            self.config.min_impurity_decrease_max,
            log=True
        )

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            max_samples=max_samples,
            max_features=max_features,
            min_impurity_decrease=min_impurity_decrease,
            **self.config.clf_params)

        clf.fit(self.train_data.values, self.train_labels.values)
        val_predictions = clf.predict(self.val_data.values)
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
