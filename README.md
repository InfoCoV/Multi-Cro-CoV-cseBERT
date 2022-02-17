# Multi-Cro-CoV-cseBERT
The project explores the performance of various machine learning algorithms for the retweeting prediction problem, depending on the provided features.

## Data description

Tweets are labeled into two classes:

`0`: tweets retweeted only once

`1`: tweets retweeted more than once

Types of features:

**a)** Content features only, extracted by a transformer model `InfoCoV/Cro-CoV-cseBERT`

**b)**  Various tabular features representing Twitter users and their interactions

**c)**  Joined features a) and b)

## Installation
Tested on `Python 3.9.9`

`pip install -r requirements.txt` (virtualenv recommended)

## Data placement
Original input data: `./data/original/Org-Retweeted-Vectors_preproc.csv`

Create folders for processed data:

`./data/intermediary/`

`./data/prepared/`

## Data exploration and feature analysis
`./notebooks/exploration/features_analysis.ipynb`

## Data Processing and preparation
Run scripts in the preparation folder:

1. `bert_extract.py` to extract content features
2. `tran_val_test.py` to create train, validation, and test splits for `id_str`
3. `feature_preparation.py` for training, validation, and test df, without content features
4. `full_feature_joing.py` for training, validation, and test df for joined features and content features alone

Alternatively, you can download all the original and prepared data from here: [data.zip](https://drive.google.com/file/d/1At1GdEStQKE9664bk8WakNSdxO3lGf1B/view?usp=sharing)

After download, unzip all to `data`.

## Baseline training runs
Notebooks in the `notebooks/baselines` folder investigate MLP and Random Forest runs:

1. `content_only_2_labels.ipynb`, content features only
2. `features_only_2_labels`, network and tabular features without content features
3. `full_features_2_labels`, all features

## Model training
Run `train.py "some_config.yaml"` by providing one of the configs in the `configs` folder.

The config file specifies the model type and all related parameters.

The training script runs multiple experiments to find the best hyperparameters by using [Optuna](https://optuna.org/) optimization.

Results and model checkpoints are stored in the `experiments` folder.

You can download all results and model checkpoints from here: [experiments.zip](https://drive.google.com/file/d/1qI3FOtujREo7r17SGrlVE3XlFQ3DWG7X/view?usp=sharing)

After download, unzip all to `experiments`.

## Results analysis

Notebooks in `notebooks/results_analysis` analyze all the results from `experiments`.
