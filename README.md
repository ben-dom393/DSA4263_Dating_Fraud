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

- Cd into directory
```bash
cd DSA4263_Dating_Fraud
```

---

It is recommended to set up a virtual environment or use docker to replicate the project.
### Virtual environment
1. Create virtual environment
``` bash
python3 -m venv .venv
```
2. Activate the virtual environment
``` bash
source .venv/bin/activate
```
3. Install required packages and dependencies
``` bash
$ pip install -r requirements
```
4. cd into src directory
``` bash
cd src
```
5. Run the prediction
``` bash
python models/final_model.py <path/to/dataset> [--output_path path/to/output/dir/]
```

eg.
``` bash
python models/final_model.py ../data/processed/final_test_dataset.csv --output_path ./
```



## Directory structure

```
├── LICENSE
├── README.md       
├── data
│   ├── interim
│   ├── processed
│   └── raw
├── models
├── notebooks
├── references
├── requirements.txt
└── src               
    ├── __init__.py
    ├── data
    │   └── make_dataset.py
    ├── features
    │   └── build_features.py
    ├── models       
    │   ├── predict_model.py
    │   └── train_model.py
    └── visualization
        └── visualize.py

```
