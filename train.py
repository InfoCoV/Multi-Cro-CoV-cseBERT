"""Runs training experiments"""

import argparse
import importlib
import logging

from omegaconf import OmegaConf

LOGGER = logging.getLogger(__name__)


def train(config):
    LOGGER.info(f"Running model {config.model_module}")
    model_module = importlib.import_module(config.model_module)
    model_module.config = OmegaConf.merge(model_module.config, config)
    LOGGER.info("Full config:")
    LOGGER.info(model_module.config)

    model_module.run(model_module.config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to YAML config")

    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    LOGGER.info("Loaded yaml config file:")
    LOGGER.info(config)

    train(config)
