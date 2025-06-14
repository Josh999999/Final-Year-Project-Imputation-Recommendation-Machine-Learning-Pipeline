# Imports

from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import json




"""# Pipeline section 1 - Input (target database)

In this section we take user input in the form of a file. This section of the pipeline will eventually evolve into a visual section in the user interface, instead here there will be a pop-up that simply gets the user to select a file from the users machines file explorer
"""

def get_file(file_path) -> pd.DataFrame:
    file_obj = Path(file_path.name)
    df = None


    # Best not to do a series of embedded if statements, probably change to a switch case down line
    if file_obj.suffix == ".csv":
        df = pd.read_csv(file_path, skipinitialspace=True, converters={col: str.strip for col in range(0, 10)})

    
    elif file_obj.suffix == ".json":
        #df = pd.read_json(file_path).applymap(lambda x: x.strip() if isinstance(x, str) else x)

        # Read the entire file content as a string
        json_string = file_path.read().decode("utf-8")


        # Try to decode the entire string as a single JSON object
        try:
            json_data = json.loads(json_string)
            
        except json.JSONDecodeError:

            # If decoding as a single object fails, assume it's a sequence of objects and decode them one by one
            json_data = [json.loads(line) for line in json_string.splitlines() if line.strip()]


        # Assuming your JSON file is a list of dictionaries
        df = pd.DataFrame(json_data)
        print(df)


    # Check file is not empty
    if df.empty:


        raise ValueError(f"Given File: {file_path.name}, file is empty")


    return df




"""# Pipeline section 2 - Dataset preprocessing and cleaning

In this section we take the user inputted file from the previous input section of the pipeline. Here the file is cleaned (for example by removing any data points of mismatching data types from non-missing values columns or by removing any improper values such as inf from all columns) Data also needs to be cleaned by removing missing values from the non missingness columns as by this section the user should have chosen a missingness column they want the algorithm reccomendation for
"""

def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    # Find missing column
    missingness_columns = df.columns[df.isnull().any()].tolist()


    if len(missingness_columns) < 1:


        raise KeyError("The missingness dataset must have at least one column with missing values")
    

    missingness_column = missingness_columns[0]
    

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()


    # Strip whitespace from string values in each cell
    for col in df.select_dtypes(include=['object']):
        df[col] = df[col].str.strip()


    # # Drop rows with all missing values
    # independant_columns = [col for col in df.columns if col != missingness_column]
    # df = df.dropna(subset=independant_columns, axis=0, how='all')


    # Replace any numeric values that arent real numbers
    for column in [column for column in df.columns if not column == missingness_column]:


        if pd.api.types.is_numeric_dtype(df[column]):
            df = df[np.isfinite(np.array(df[column]))]


    # Finally remove duplicate rows
    df = df.drop_duplicates()


    return df.reset_index(), missingness_column




"""# Pipeline section 3 - Dataset metafeatures extraction

In this section we take the dataset from the user inmputted section that has now been cleaned / preproccessed in the previous section and extract the metafeatures and relevant information from the dataset that will be used by the machine learning model in the subsequent section. The subsections of this section will explain the different types of metafeatures extracted that are separated into individual function


#### Statistical metafeatures

This section contains functionality to extract the statical metafeatures from the cleaned / pre-proccessed dataset. These metafeatures are taken from a reference paper and contained in the following table
"""

def get_statistical_metafeatures(df: pd.DataFrame, missingness_column: str) -> dict[str, int]:
    """
    Return a dict of metafeatures under the typical 'simple' category
    """

    mf_simple = {}



    # Metafeatures not in the table

    # Number of Samples (rows)
    mf_simple["n_samples"] = float(len(df)) # Check this is best way of getting rows

    # Number of continuous features
    mf_simple["n_cols_continuous"]  = float(len(df.select_dtypes(include=["int64", "float64"]).columns))

    # Number of categorical features
    mf_simple["n_cols_categorical"] = float(len(df.select_dtypes(include=["object", "category"]).columns))

    # Log number of features over samples
    mf_simple["log_n_features_over_n_samples"] = float(np.log(len(df.columns)/len(df)))



    # Metafeature in the table

    # Number of patterns
    mf_simple["n_patterns"] = float(df.shape[0])

    # Log number of patterns
    mf_simple["log_n_patterns"] = float(np.log(df.shape[0]))

    # Number of Features (cols)
    mf_simple["n_features"] = float(df.shape[1])

    # Log number of features
    mf_simple["log_n_features"] = float(np.log(df.shape[1]))

    # Number of patterns with missing values
    mf_simple["n_patterns_with_missing"] = float(df.isna().any(axis=1).sum())

    # Percentage of patterns with missing values
    mf_simple["p_patterns_with_missing"] = float(df.isna().any(axis=1).sum() / df.shape[0] * 100)

    # Number of features with missing values
    mf_simple["n_features_with_missing"] = float(df.isna().any(axis=0).sum())

    # Percentage of features with missing values
    mf_simple["p_features_with_missing"] = float(df.isna().any(axis=0).sum() / df.shape[1] * 100)

    # Number of missing values
    mf_simple["n_missing_values"] = float(df.isna().sum().sum())

    # Percentage of missing values
    mf_simple["p_missing_values"] = float(df.isna().sum().sum() / (df.shape[0] * df.shape[1]) * 100)

    # Number of numeric features
    mf_simple["n_numeric_features"] = float(df.select_dtypes(include=["int64", "float64", "number"]).shape[1])

    # Number of categorical features
    mf_simple["n_categorical_features"] = float(df.select_dtypes(include=['object', 'category']).shape[1])

    # Ratio of numeric features to categorical features
    mf_simple["r_continuous_categorical"] = float(mf_simple["n_numeric_features"] / mf_simple["n_categorical_features"] if not mf_simple["n_categorical_features"] == 0 else mf_simple["n_numeric_features"])

    # Ratio of categorical features to numeric features
    mf_simple["n_categorical_continuous"] = float(mf_simple["n_categorical_features"] / mf_simple["n_numeric_features"] if not mf_simple["n_numeric_features"] == 0 else mf_simple["n_categorical_features"])

    # Dimensionality of the dataset
    mf_simple["d_dataset"] = float(sum(_ for _ in df.shape))

    # Log dimensionality of the dataset
    mf_simple["log_d_dataset"] = float(np.log(sum(_ for _ in df.shape)))

    # Inverse dimensionality of the dataset
    mf_simple["inv_d_dataset"] = float(1 / sum(_ for _ in df.shape))

    # Log inverse dimensionality of the dataset
    mf_simple["log_inv_d_dataset"] = float(np.log(1 / sum(_ for _ in df.shape)))

    # Get the average variance from numeircal columns
    mf_simple["variance"] = float(np.mean([df[column].var() for column in df.select_dtypes(include=["int64", "float64"]).columns]))

    # Get the average mean from numeircal columns
    mf_simple["mean"] = float(np.mean([df[column].mean() for column in df.select_dtypes(include=["int64", "float64"]).columns]))



    return mf_simple


"""#### Missingness metafeatures

This section contains functionality to extract the missingness metafeatures  from the cleaned / pre-proccessed dataset. These metafeatures include information about the missing data such as the proportion of missing data the missing data type, missing data mechansim, etc
"""

def get_missingness_metafeatures(df: pd.DataFrame, missingness_column, missingness_mechansim) -> dict[str, int]:
    """
    Return a dict of metafeatures
    """

    missingness_mechansim_map = {
        "MCAR": 0,
        "MAR": 1,
        "MNAR": 2
    }

    mf_missingness = {}


    # Create copy of the dataframe
    df_copy = df.copy()

    encoder = LabelEncoder()


    # Encode column if is categorical
    if not pd.api.types.is_numeric_dtype(df_copy[missingness_column]):
        df_copy[missingness_column] = encoder.fit_transform(df_copy[missingness_column])
        mf_missingness["is_numeric"] = float(0) #(False)

    else:
        mf_missingness["is_numeric"] = float(1) #(False)


    # Encode other columns if neccessary
    for column in df_copy.columns:


        if not pd.api.types.is_numeric_dtype(df_copy[column]) and not column == missingness_column:
            df_copy[column] = encoder.fit_transform(df_copy[column])


    # Get encoded missingness mechansim
    mf_missingness["missingness_mechansim"] = float(missingness_mechansim_map[missingness_mechansim.strip().upper()])




    # Metafeature extraction - General statistics

    # Mean of the missingness column
    # mf_missingness["mean"] = float(df_copy[missingness_column].mean())

    # # Variance of the missingness column
    # mf_missingness["variance"] = float(df_copy[missingness_column].var())

    # Median of the missingness column
    mf_missingness["median"] = float(df_copy[missingness_column].median())

    # Mode of the missingness column
    mf_missingness["mode"] = float(df_copy[missingness_column].mode().iloc[0])

    # Standard deviation of the missingness column
    mf_missingness["std"] = float(df_copy[missingness_column].std())

    # Minimum of the missingness column
    mf_missingness["min"] = float(df_copy[missingness_column].min())

    # Maximum of the missingness column
    mf_missingness["max"] = float(df_copy[missingness_column].max())

    # Range of the missingness column
    mf_missingness["range"] = float(df_copy[missingness_column].max() - df_copy[missingness_column].min())

    # Skewness of the missingness column
    mf_missingness["skewness"] = float(df_copy[missingness_column].skew())

    # Kurtosis of the missingness column
    mf_missingness["kurtosis"] = float(df_copy[missingness_column].kurt())

    # Interquartile range of the missingness column
    mf_missingness["iqr"] = float(df_copy[missingness_column].quantile(0.25) - df_copy[missingness_column].quantile(0.75))



    # Metafeature extraction - Data quality

    # Number of unqiue values in the missingness column
    mf_missingness["n_unique_values"] = float(df_copy[missingness_column].nunique())

    # Constant of the missingness column
    mf_missingness["constant"] = float(len(df[df_copy[missingness_column].isin([df_copy[missingness_column].mode()])]) / df_copy.shape[0])



    # Metafeature extraction - Entropy & Variability

    # Shannon entropy of the missingness dataset
    mf_missingness["shannon_entropy"] = float(-np.sum(df_copy.value_counts(normalize=True) * np.log2(df_copy.value_counts(normalize=True))))

    # Coefficient of variation of the missingness dataset
    mf_missingness["coefficient_of_variation"] = float(np.mean(df_copy.std()) / np.mean(df_copy.mean()))

    # Signal-to-noise ratio of variation of the missingness dataset
    mf_missingness["signal_to_noise_ration"] = float(np.mean(df_copy.mean()) / np.mean(df_copy.std()))



    # Metafeature extraction - Missingness stats

    # # Number of missing values in the missingness column
    # mf_missingness["n_missing_values"] = float(len(list(df_copy[missingness_column].isna())))

    # # Missingness ration of the missingness column
    # mf_missingness["p_missing_values"] = float(len(list(df_copy[missingness_column].isna())) / len(list(df_copy[missingness_column])) * 100)




    return mf_missingness




"""# Pipeline section 4 - Imputation algorithm recommendation machine learning model

In this section we take the metafeatures that have been extracted from the cleaned / pre-proccessed dataset inputted by the user and run them through the pre-trained multiclassification imputation reccomendation mechine learning model. This model has already be trained on the specific metafeatures extracted from the dataset to given a "reccomendation" of the optimal imputation algorithm to fill in the missing data points in the dataset the metafeatures were extracted from
"""

def model_recommendation(X_input, model_path, encoded_classess_path):
  # Load the saved model
  loaded_model = joblib.load(model_path)


  # Load the classess encoding
  with open(encoded_classess_path, 'r') as f:
      encoded_classess = np.array(json.loads(f.read()))
      
  print("\n\n\n\n", encoded_classess, "\n\n\n\n")


  # Predict with loaded model
  y_pred = loaded_model.predict(X_input)


  return encoded_classess[y_pred][0]