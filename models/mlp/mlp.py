"""MLP model definition and experiment setup"""

import logging

from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from omegaconf import OmegaConf
import optuna
from optuna.samplers import TPESampler

from optuna_utils import OptunaExperiments, run_experiments


LOGGER = logging.getLogger(__name__)

# updated by train.py before running
config = OmegaConf.create(
    {"clf_params": {
        "max_iter": 50, "random_state": 1, "verbose": True,
        "early_stopping": True, "validation_fraction": 0.1,
        "n_iter_no_change": 5},
     "num_layers_min": 1,
     "num_layers_max": 3,
     "num_hidden_exp_min": 5,
     "num_hidden_exp_max": 10,
     "lr_exp_min": -4,
     "lr_exp_max": -2,
     "lr_alpha_min": -4,
     "lr_alpha_max": -2,
     "batch_exp_min": 5,
     "batch_exp_max": 8}
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
        num_layers = trial.suggest_int(
            "num_layers",
            self.config.num_layers_min, self.config.num_layers_max)
        hidden_layer_sizes = []
        for i in range(1, num_layers + 1):
            size = 2 ** trial.suggest_int(
                f"num_hidden_exp_{i}",
                self.config.num_hidden_exp_min,
                self.config.num_hidden_exp_max)
            hidden_layer_sizes.append(size)
        lr_exp = trial.suggest_int(
            "lr_exp", self.config.lr_exp_min, self.config.lr_exp_max)
        lr = 10 ** lr_exp
        alpha_exp = trial.suggest_int(
            "alpha_exp", self.config.lr_alpha_min, self.config.lr_alpha_max)
        alpha = 10 ** alpha_exp
        batch_exp = trial.suggest_int(
            "batch_exp", self.config.batch_exp_min, self.config.batch_exp_max)
        batch_size = 2 ** batch_exp

        clf = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            learning_rate_init=lr,
            batch_size=batch_size,
            alpha=alpha,
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
