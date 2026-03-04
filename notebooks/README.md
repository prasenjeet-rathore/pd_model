# Modeling Pipeline & Notebooks

## 1. Data Preparation
File: 1_data_preparation.ipynb

Scope: Initial ingestion and filtering of the  dataset.

Key Tasks: Filtering for Estonian customers, handling data types, and performing initial exploratory data analysis (EDA) to ensure data quality.

## 2. Target Creation
File: 2_target_creation.ipynb

Scope: Defining the business-critical "Default" event.

Key Tasks: Implementing logic and creating the target variable used for in modeling.


## 3. Feature Engineering & WOE

File: 3_feature_engineering.ipynb

Scope: Transforming raw variables into predictive signals.

Key Tasks: Perform feature engineer , engineering new variables , check correlation, drop variables and bin variables for modeling.


## 4. Modeling & Evaluation

File: 04_modeling.ipynb

Scope: Model training, evaluation, calibration, , tune selected model and export model.

Key Tasks: create two models Logistic Regression and xgboost, evaluate performance via AUC, pick one model and then export models.



### IMPORTANT
to run these notebooks the pre-requisites is to install required libraries from requirements.txt in root folder. 
shap library can cause some issues if you are missing c++ compiler on your setup. 

Ideal setup would be to have docker-compose for dev environment but due to time-constraint on the project currently this approach is chosen. 