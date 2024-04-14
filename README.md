# Overview

*add real and AI images and example of profile*

## Analysis

## Models

## Features

## Results

## 

## Getting started
1. Install Docker: 
    - For Mac: https://store.docker.com/editions/community/docker-ce-desktop-mac
    - For Windows: https://store.docker.com/editions/community/docker-ce-desktop-windows
    - For Linux: Go to this page and choose the appropriate install for your Linux distro: https://www.docker.com/community-edition
        - Install Docker Compose (https://docs.docker.com/compose/install/#install-compose):
            ```bash
            $ sudo curl -L https://github.com/docker/compose/releases/download/1.21.0/docker-compose-$(uname -s)-$(uname -m) -o /usr/local/bin/docker-compose
            ```
            ```bash
            $ sudo chmod +x /usr/local/bin/docker-compose
            ```
            Test the installation:
            ```bash
            $ docker-compose --version
            docker-compose version 1.21.0, build 1719ceb
            ```
2. Install 
    ``` bash
    $ pip install -r requirements
    ```
    It is recommended to set up a central virtualenv or condaenv for cookiecutter and any other "system" wide Python packages you may need.



## Directory structure

```
├── LICENSE
├── Makefile           <- Makefile with commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.py           <- Make this project pip installable with `pip install -e`
└── src                <- Source code for use in this project.
    ├── __init__.py    <- Makes src a Python module
    │
    ├── data           <- Scripts to download or generate data
    │   └── make_dataset.py
    │
    ├── features       <- Scripts to turn raw data into features for modeling
    │   └── build_features.py
    │
    ├── models         <- Scripts to train models and then use trained models to make
    │   │                 predictions
    │   ├── predict_model.py
    │   └── train_model.py
    │
    └── visualization  <- Scripts to create exploratory and results oriented visualizations
        └── visualize.py
```
