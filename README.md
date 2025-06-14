# Final-Year-Project-Imputation-Recommendation-Machine-Learning-Pipeline
This is my Final Year Project I developed as the final peace of coursework for my University Degree

## Identified problem / issue to solve
Missing data is a common occurrence in research and can have a negative impact on statistical analysis. Even in controlled studies that are designed to reduce occurrences of negatives impacts like loss, missing data occurs in most research, missingness can lead to incorrect or invalid conclusions by producing estimations that are biased (Kang, H., 2013).

Imputation is an effective technique for handling missing data by replacing missing values with estimations based on the remaining data. The are multiple imputation algorithms and there are many different factors to consider when determining the optimal algorithm such as; missingness pattern, missingness mechanism, data complexity, etc. Analysis can be performed on the dataset to extract features (such as the missingness ratio, co-variance, etc) which can be used to determine which imputation methods should be considered either by the user or a mathematical or non-mathematical formulated system. The problem with this is it is all too likely that in a large portion of fields of research, researchers will not be familiar with either imputation methods or how to perform statistical analysis on a dataset. This can be shown from a study performed analysing how other research studies handled the issue of missing data with imputation and found that out of 58 none of the studies considered both the mechanism and pattern of missingness when selecting an imputation algorithm (Afkanpour, M. et al, 2024) 


## Proposed solution (artefact)
This primary solution this project aims to implement to address the problems stated previously is to train a machine learning model to take in a datasets metafeatures and use them to give a “recommendation” of the best imputation algorithm for this dataset.

This machine learning model will then become part of a machine learning pipeline that; takes in a dataset, preprocesses the data, passes the datasets features through the machine learning model, and outputs the imputation algorithm “recommendation”. This tool would come with a user interface making it incredibly useful for any researcher who may have missing data in their research but not the technical capabilities to identify the optimal imputation methodology to replace their missing data points.


## Project aims
Due to research showing that it is not feasible to create any kind of model to determine the missingness mechanism of a dataset using only said dataset we have refocused the aims and objectives from those stated in appendix A.
-	**Aim 1** - Create functionality to generate missing data datasets and record their optimal imputation algorithm for use when training the machine learning model
-	**Aim 2** - Create and train a machine learning pipeline to “recommend” the optimal imputation algorithm for a given dataset
-	**Aim 3** - Create a GUI for the pipeline tool


## Content of the repository:
  - **Python (Flask)** - This runs the machine learning model and the pipeline. It also uses the Streamlit library / web framework to create the single page application that serves as the GUI for the pipeline

---
## NoteBooks
The first parts of this project are the Jupiter notebooks (reccomended to run with Google Collab althought without the file structure required they won't run properly). These notbooks are used to document and create functionality which generates key elements of this project such as:
  - **Data gathering** - Connecting to the Kaggle API (Online web repository for datasets) to retrevie datasets that can be augmented and use to train the machine learning model
  - **Missing data generation** - Algorithms to generate missing data in the retreived datasets for each type of missingness mechnasim
  - **Imputation algorithms and assessment** - Runs imputation algorithms on each datasets with missing data to find the best imputation algorithm for each dataset (along with extracting their metafeatures with are used to train the classification model with the optimal imputation algorithm as the label and metafeatures as input) which is then stored
  - **Model training and optimisation** - Training of the classification model on the data generated, and further optimisation of the model using parameter adjustment and hyperparameter searching


## Pipeline
This part of the project takes the machine learning model trained and optimised from the notbooks and uses it to create the pipeline. This is a functional pipeline which takes in a users given dataset (with missing data), cleans its, extracts its metafeatures and uses them as input to get a "reccomendation" of the optimal imputation algorithm from the machine learning model (this "reccomendation" is the models predicition as to which imputation algorithm will best replace the missing data in the dataset)

---

## Running the site
To run the web application in it's current state you will need to follow the current steps


### Step 1. Download Fask
Download the current python version of flask using pip from the command line
```
python3 -m pip install Flask
```
<br>


### Step 2. Run the web application using Streamlit
Run the web application using the Streamlit web framework
```
streamlit run pipeline_view.py
```
<br>


