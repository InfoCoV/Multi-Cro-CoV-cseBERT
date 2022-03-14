"""
TabNet model definition and experiment setup.

TabNet: Attentive Interpretable Tabular Learning
https://arxiv.org/abs/1908.07442

Model details:
https://pytorch-tabular.readthedocs.io/en/latest/models/#tabnet
"""

import logging
import os.path
import shutil

from sklearn.metrics import classification_report
from omegaconf import OmegaConf
import optuna
from optuna.samplers import TPESampler
from pytorch_tabular import TabularModel
from pytorch_tabular.models import TabNetModelConfig
from pytorch_tabular.config import (
    DataConfig, OptimizerConfig, TrainerConfig, ExperimentConfig)
from pytorch_tabular.utils import get_class_weighted_cross_entropy

from optuna_utils import OptunaExperiments, run_experiments


LOGGER = logging.getLogger(__name__)
LABEL_COL = "retweet_label"

# updated by train.py before running
config = OmegaConf.create(
    {"max_epochs": 50,
     "lr_exp_min": -4,
     "lr_exp_max": -3,
     "alpha_exp_min": -4,
     "alpha_exp_max": -3,
     "batch_exp_min": 5,
     "batch_exp_max": 8,
     "n_d_exp_min": 2,
     "n_d_exp_max": 6,
     "n_a_exp_min": 2,
     "n_a_exp_max": 6,
     "n_steps_min": 3,
     "n_steps_max": 10,
     "gamma_min": 1.0,
     "gamma_max": 2.0,
     "categorical_cols": [
         "entities.urls", "entities.media", "user_in_net",
         "has_covid_keyword", "user.followers_isna",
         "users_mention_isna", "following_users_isna",
         "users_reply_isna"],
     "exp_log_freq": 100,
     "seed": 1,
     "num_workers": 24}
)


class Experiments(OptunaExperiments):

    def __init__(
            self,
            train_data,
            val_data,
            train_labels,
            val_labels,
            experiment_root,
            config):

        self.train_data_joined = train_data.copy()
        self.train_data_joined[LABEL_COL] = train_labels
        self.val_data_joined = val_data.copy()
        self.val_data_joined[LABEL_COL] = val_labels

        self.experiment_root = experiment_root
        self.config = config

        self.study = self.create_study()
        self.best_score = None

        self.cat_col_names = config.categorical_cols
        self.num_col_names = [
            c for c in train_data.columns if c not in config.categorical_cols]
        self.data_config = DataConfig(
            target=[LABEL_COL],
            continuous_cols=self.num_col_names,
            categorical_cols=self.cat_col_names,
            normalize_continuous_features=False,
            num_workers=config.num_workers)

        self.weighted_loss = get_class_weighted_cross_entropy(
            train_labels.values.ravel(), mu=0.1)

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
        lr_exp = trial.suggest_int(
            "lr_exp", self.config.lr_exp_min, self.config.lr_exp_max)
        lr = 10 ** lr_exp
        alpha_exp = trial.suggest_int(
            "alpha_exp", self.config.alpha_exp_min, self.config.alpha_exp_max)
        alpha = 10 ** alpha_exp
        batch_exp = trial.suggest_int(
            "batch_exp", self.config.batch_exp_min, self.config.batch_exp_max)
        batch_size = 2 ** batch_exp

        n_d_exp = trial.suggest_int(
            "n_d_exp", self.config.n_d_exp_min, self.config.n_d_exp_max)
        n_d = 2 ** n_d_exp
        n_a_exp = trial.suggest_int(
            "n_a_exp", self.config.n_a_exp_min, self.config.n_a_exp_max
        )
        n_a = 2 ** n_a_exp
        n_steps = trial.suggest_int(
            "n_steps", self.config.n_steps_min, self.config.n_steps_max)
        gamma = trial.suggest_float(
            "gamma", self.config.gamma_min, self.config.gamma_max)

        experiment_path = self.config.experiment_root
        checkpoints_path = os.path.join(experiment_path, "checkpoints")
        tb_logs = os.path.join(experiment_path, "tb_logs")
        run_name = "tabnet"

        # store all just for the current optuna run
        if os.path.exists(checkpoints_path):
            shutil.rmtree(checkpoints_path)
        if os.path.exists(tb_logs):
            shutil.rmtree(tb_logs)

        trainer_config = TrainerConfig(
            auto_lr_find=False,
            gpus=1,
            deterministic=True,
            batch_size=batch_size,
            max_epochs=self.config.max_epochs,
            checkpoints_path=checkpoints_path,
        )

        optimizer_config = OptimizerConfig(
            optimizer="AdamW",
            optimizer_params={"weight_decay": alpha}
        )

        model_config = TabNetModelConfig(
            task="classification",
            learning_rate=lr,
            loss=self.weighted_loss,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma
        )

        experiment_config = ExperimentConfig(
            project_name=tb_logs,
            run_name=run_name,
            exp_log_freq=self.config.exp_log_freq
        )

        tabular_model = TabularModel(
            data_config=self.data_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            trainer_config=trainer_config,
            experiment_config=experiment_config
        )

        tabular_model.fit(
            train=self.train_data_joined,
            validation=self.val_data_joined,
            seed=self.config.seed,
            loss=self.weighted_loss)

        result = tabular_model.evaluate(self.val_data_joined)
        LOGGER.info(result)
        pred_df = tabular_model.predict(self.val_data_joined)
        val_predictions = pred_df.prediction.values

        out = classification_report(
            self.val_data_joined[LABEL_COL].values, val_predictions,
            digits=3, output_dict=True)
        LOGGER.info(out)
        f1 = out["macro avg"]["f1-score"]

        if self.best_score is None or f1 > self.best_score:
            self.best_score = f1
            self.store_results(tabular_model, out)
            self.store_study()

        return f1


def run(config):
    run_experiments(
        config=config,
        experiments_class=Experiments)
