from abc import ABC, abstractmethod
import logging
import os
from pathlib import Path

import joblib
from omegaconf import OmegaConf

from data_utils import setup_data


LOGGER = logging.getLogger(__name__)


# https://optuna.readthedocs.io/en/stable/faq.html#how-can-i-output-a-log-only-when-the-best-value-is-updated
def logging_callback(study, frozen_trial):
    previous_best_value = study.user_attrs.get("previous_best_value", None)
    if previous_best_value != study.best_value:
        study.set_user_attr("previous_best_value", study.best_value)
        LOGGER.info(
            f"Trial {frozen_trial.number} finished"
            f"with best value: {frozen_trial.value}"
            f"and parameters: {frozen_trial.params}")


class OptunaExperiments(ABC):

    def __init__(
            self,
            train_data,
            val_data,
            train_labels,
            val_labels,
            experiment_root,
            config):

        self.train_data = train_data
        self.val_data = val_data
        self.train_labels = train_labels
        self.val_labels = val_labels
        self.experiment_root = experiment_root
        self.config = config

        self.study = self.create_study()
        self.best_score = None

    @abstractmethod
    def create_study(self):
        """Creates Optuna study"""
        pass

    @abstractmethod
    def optimize(self):
        """Runs Optuna optimization"""
        pass

    @abstractmethod
    def objective(self, trial):
        """Optuna objective function to optimize"""
        pass

    def store_study(self):
        LOGGER.info(f"Saving study to {self.experiment_root}")
        joblib.dump(self.study, Path(self.experiment_root) / "study.pkl")

    def store_results(self, model, results):
        LOGGER.info(f"Saving best results to {self.experiment_root}")
        joblib.dump(model, Path(self.experiment_root) / "model.pkl")
        conf = OmegaConf.create(results)
        OmegaConf.save(conf, Path(self.experiment_root) / "results.yaml")


def run_experiments(config, experiments_class):
    experiment_root = Path(config.experiment_root)
    os.makedirs(experiment_root, exist_ok=True)
    data_dict = setup_data(config)

    if data_dict["scaler"] is not None:
        LOGGER.info(f"Storing data scaler to {config.experiment_root}")
        joblib.dump(data_dict["scaler"], experiment_root / "scaler.pkl")

    LOGGER.info("Starting Optuna runs")
    experiments = experiments_class(
        train_data=data_dict["train_data"],
        val_data=data_dict["val_data"],
        train_labels=data_dict["train_labels"],
        val_labels=data_dict["val_labels"],
        experiment_root=config.experiment_root,
        config=config
        )
    experiments.optimize()
