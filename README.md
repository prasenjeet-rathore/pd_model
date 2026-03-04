# PD MODEL

<p align="center">
  <a href="https://cookiecutter-data-science.drivendata.org/"><img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" /></a>
  <img src="https://img.shields.io/badge/SciPy-%230C55A5.svg?style=flat&logo=scipy&logoColor=white" />
  <img src="https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=flat&logo=Matplotlib&logoColor=black" />
  <img src="https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi" />
  <img src="https://img.shields.io/badge/jupyter-%23FA0F00.svg?style=flat&logo=jupyter&logoColor=white" />
  <img src="https://img.shields.io/badge/docker-%230db7ed.svg?style=flat&logo=docker&logoColor=white" />
</p>

The following is a project for experimentation on PD model for credit risk modeling
By Prasenjeet Rathore

## Project Organization

```
├── LICENSE            <- Open-source license MIT
├── README.md          <- The top-level README for using this project.
├── app
│   └── app.py         <- Fast-Api App to use saved model as an endpoint.
├── data
│   ├── 01_raw            <- The original, immutable data dump.
│   ├── 02_processed      <- Intermediate data that has been transformed.
│   └── 03_final          <- The final data sets for modeling.
│
├── docs               <- A default docs folder
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                      
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         src and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures from notebooks
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `uv pip freeze > requirements.txt`
│
└── src   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── modeling                
    │   ├── __init__.py         <- Makes modeling a Python module
    │   ├── modeling.py         <- Code to run model pipeline        
    │   └── test_prediction.py  <- Code to run prediction using one row from out of time sample
    │
    ├── utils 
    │   ├── __init__.py         <- Makes util a Python module
    │   │── config.py           <- Store useful variables and configuration
    │   │── data_cleaning.py    <- util functions for data processing
    │   │── evaluation.py       <- util functions for evaluating models
    │   │── features.py         <- util functions to create features for modeling
    │   │── target.py           <- util functions to create target variable for                    
    │   └── woe.py              <- util function for woe and binning
    │
    └── 
```


## 1. How to to test the model prediction?

This repository includes a two-stage `Dockerfile` at the project root. It uses:

- **Builder stage** – installs dependencies 
- **Runtime stage** – copies the virtual environment and project code into a slim Python image.

### Build the image 
*(use either docker or podman, I used podman for my fedora workstation)*

*Build may takes 2-3 minutes or maybe more becuase it is 2 stage build to avoid attack surface for app and size* 

### 1.1 From the project root run:

```bash
docker build -t model-pd .
```

### 1.2 Run the container in detached mode

```bash
docker run -d -p 8000:8000 model-pd
```

### 1.3 Check if container is created and running, try -a if container is not showing

```bash
docker ps
```

### 1.4 To see a one-off prediction from model run the following
The data for testing is one row picked from out of out-of-time sample 
<br> currently chosen model for testing is logistic regression


```bash
docker run --rm model-pd python -m src.modeling.test_prediction
```

#### 1.5 How to stop container

first check the contatiner id by running this
```bash
docker ps
```

then
```bash
docker stop *insert container-id*
```

 *small tip, you need to write full container id or copy paste it just write few first characters then hit tab , it will autocomplete*

## How to run notebooks ?

For using the notebook considering install packages with requirements.txt as requirements-prod.txt is solely for production container to keep container size small. Make sure to install it in a .venv folder setup at the root of the folder. 

also for setting up dev environment make sure to setup the python interpreter path in your choice of ide, sometimes jupyter notebooks otherwise don't detect the .venv created. (in Vscode the method is Ctrl+Shift+P then select Python Interpreter)

I recommend setting up environment using uv

--------