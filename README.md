# Overview

*add real and AI images and example of profile*

## Getting started

- Clone this repository

*through https* or
``` python
git clone https://github.com/ben-dom393/DSA4263_Dating_Fraud.git
```
*through ssh*
```python
git clone git@github.com:ben-dom393/DSA4263_Dating_Fraud.git
```

- Cd into the directory
```bash
cd DSA4263_Dating_Fraud
```

---

It is recommended to set up a virtual environment or use docker to replicate the project.
### Docker
1. Build the docker image
``` bash
docker build -t dsa4263 .
```
2. Run a container from the image
``` bash
docker run dsa4263
```

### Virtual environment
1. Create virtual environment
``` bash
python3 -m venv datingfraud
```
2. Activate the virtual environment
``` bash
source datingfraud/bin/activate
```
3. Install required packages and dependencies
``` bash
$ pip install -r requirements
```
    



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
