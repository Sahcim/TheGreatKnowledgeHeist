# TheGreatKnowledgeHeist

## Prerequisites
```
python3.9
```
## Installation
### 1. Virtual enviroment
Set up python virtualenv, e.g using Python's venv module:
```
python3 -m venv venv_dirname
source venv_dirname/bin/activate
```
### 2. Install Project
Install project:
```
pip install -e .
```

## Datasets
To create datasets used in experiments use:


### Swag
```
python3 scripts/download_and_prepare_dataset.py swag data --sample_train 50000 --sample_validation 5000
```
### Amazon
```
python3 scripts/download_and_prepare_dataset.py amazon_polarity data --sample_train 50000 --sample_validation 5000
```
### Acronyms
```
python3 scripts/download_and_prepare_dataset.py acronym_identification data
```