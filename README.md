# Final-Year-Project-Imputation-Recommendation-Machine-Learning-Pipeline
This is my Final Year Project I developed as the final peace of coursework for my University Degree

### Content of the repository:
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


