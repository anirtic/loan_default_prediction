# Loan Default and Early Re-Payment Prediction

This project aims to predict default and early re-payment behaviours from loan applications based on various features such as applicant information, loan amount, credit score, and engineered features.

## Project Files and Folders
- `app/` Folder containing code used for data preprocessing and deployment.
  - `config.py`: Folder contains paths to access the data, path to be validated to access GCP Bucket, dtypes of data we are going to read, dictionaries of aggregations we are going to make.
  - `functions.py` External functions used in analysis and deployment.
  - `functions_visualizatoin.py` External functions used in analysis for visualizations.
    - `data_extraction/`
      - `data_extractor_pipe.py`: Python file which contains various classes that are used in the Pipeline.
      - `scripts/`: Folder contains script files for each table to collect/clean/prepare data.
- `notebooks/` Folder containing machine learning models in .pkl format, as well as data preprocessing components such as scaler. Also, includes a FastAPI application and Dockerfile for creating a docker image.
  - `1 - Introduction and General Check.ipynb`: First meeting with the data, checking what data do we have on surface level.
  - `2 - Project plan and EDA.ipynb`: Brief plan of how we will conduct this analysis. Explaratory data analysis with visual graphs.
  - `3 - Statistical Inference.ipynb`: Notebook contains used statistical tests.
  - `4 - Default Risk Modelling.ipynb`: In this notebook we will conduct feature elimination, hyperparameting and will create a model to predict default risk. 
  - `5 - Early payment behavior modelling.ipynb`: In this notebook we will conduct feature elimination, hyperparameting and will create a model to predict early re-payment risk. 
- `test/`: Contains test data and notebook for HTTP POST request testing.

#### Usage
Running the Jupyter Notebook 
1. Open the .ipynb Jupyter Notebooks located on `notebooks/`.
2. Follow my path of how I, conducted data analysis, created machine learning models.

#### Using the FastAPI Application
Deployed version is able to handle application data in 3 ways, accepting HTTP POST requests of single JSON lines / reading file from GCP bucket / reading file uploaded on a browser.

Uploading file on a browser:
1. Open https://def-earl-prediction-n7pa5j5f6q-as.a.run.app/docs on a browser.
2. For default prediction model choose /default_prediction end point select button "Try it out", upload any .csv file from `test/`with and click button "Execute".
3. For early repayment behavior model choose /early_repayment_prediction end point select button "Try it out", upload any .csv file from `test/`with and click button "Execute".

POST request:
1. Open Notebook at `test/HTTP POST Requests test.ipynb` and follow the steps.

### Performance:
#### Default Prediction:
Able to handle a single request on average in 190s, resulting in 18-20 requests per hour.
#### Early Repayment prediction:
Able to handle a single request on average in 135, resulting in 26-28 requests per minute.

#### Note:
For any questions please feel free to contact me on LinkedIn/Gmail.