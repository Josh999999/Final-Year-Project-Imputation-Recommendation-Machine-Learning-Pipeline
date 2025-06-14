#"""IMPORTS"""
import streamlit as st
import os
import json
from pipeline import get_file, clean_data, get_statistical_metafeatures, get_missingness_metafeatures, model_recommendation




# Create global variables to use in the 
IMAGES = "images"
MODEL_PATH = "./model/best_random_forest_model.pkl"
ENCODED_CLASSESS_PATH = "./model/encoded_classess.json"


with open('./info/imputation_algorithms.json', 'r') as f:
    imputation_algorithms_description_lookup = json.load(f)


with open('./info/missingness_mechanisms.json', 'r') as f:
    missingness_mechansims_description_lookup = json.load(f)




#"""CODE"""
if __name__ == "__main__":
    
    # Configure the page
    st.set_page_config(
        page_title="Imputation reccomendation pipeline", 
        page_icon=str(os.path.join(IMAGES, "record-circle-outline.png")), 
        layout="wide",
        initial_sidebar_state="expanded"
    )


    st.markdown("""
    <style>
        /* General layout */
        .main { background-color: #f6f4f9; }
        section[data-testid="stSidebar"] {
            border-right: 1px solid #e0e0e0;            
            width: 110px !important;
            min-width: 110px !important;
            max-width: 110px !important;
        }
        header[data-testid="stHeader"] {visibility: hidden;}
        footer {visibility: hidden;}

        h1, h2, h3, h4 {
            font-family: 'Segoe UI', sans-serif;
        }

        .card {
            background-color: white;
            padding: 2rem;
            border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
            margin-bottom: 1rem;
        }

        .upload-section, .output-section {
            border: 1px solid #ddd;
            padding: 1rem;
            border-radius: 10px;
            background: #fff;
        }

        .uploaded-file {
            font-size: 0.9rem;
            color: #666;
        }

        /* Card styling */
        .card {
            background-color: white;
            padding: 25px;
            border-radius: 15px;
            box-shadow: 0px 2px 8px rgba(0,0,0,0.06);
        }

        .centered {
            text-align: center;
        }

        hr {
            margin-top: 10px;
            margin-bottom: 10px;
        }

        /* Upload box placeholder style */
        .upload-placeholder {
            border: 2px dashed #ccc;
            padding: 1.5rem;
            text-align: center;
            border-radius: 10px;
            color: #999;
        }
            
        .upload-box, .dropdown-box {
            background-color: #fdf3ff;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 0px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
                   
        .file-info, .dropdown-container {
            background-color: #ccc;
            padding: 12px 20px;
            border-radius: 3px;
            font-size: 14px;
            color: #111;
        }    
        .small-space {
            margin-top: 5px;
            margin-bottom: 5px;
        }
    </style>
    """, 
    unsafe_allow_html=True)


    # HEADER - Page title and tool explination
    st.header(":material/network_node: Machine learning imputation reccomendation pipeline")

    st.markdown("""
        <hr style="margin-top: 5px; margin-bottom: 5px;">
    """, unsafe_allow_html=True)

    st.write("""
        This is a tool for researchers who have data with missing values to be given an optimal imputation algorithm “recommended” by a 
        multi-class classification machine learning model trained on missing data to assess the best imputation algorithm to replace the missing data points. 
        \n
        This tool is a pipeline that takes in a dataset (with missing values), preprocesses the data, runs the processed data through a 
        trained machine learning model. this model will output an optimal imputation algorithm “recommendation” for the given dataset.
    """)


    # SIDEBAR - Immutable Sidebar Content
    with st.sidebar:

        # Display the icon as an image
        st.image("https://fonts.gstatic.com/s/i/materialicons/account_circle/v4/24px.svg", caption="Account", width=60)

        st.markdown("".join(["<br>" * 18]), unsafe_allow_html=True)

        st.markdown("""
            <hr style="margin-top: 5px; margin-bottom: 5px;">
        """, unsafe_allow_html=True)

        st.markdown("<p class='footer_elements' style='padding: 0px !important; margin: 0px !important'><b>**v1.0**</b></p>", unsafe_allow_html=True)


    # Variables for the next section
    uploaded_file = None
    uploaded_file_missingness_column = None
    uploaded_file_df = None
    uploaded_file_metafeatures = None
    imputation_algorithm_recommendation = "..."
    imputation_algorithm_recommendation_description = "..."
    dataset_warning = False
    dataset_warning_message = None
    pipline_warning = False
    pipeline_warning_message = None

    # Create input / output columns
    COLUMN_dataset_input, SPACER, COLUMN_imputation_algorithm_output = st.columns([1, 0.1, 1])


    # COLUMN - Upload the missingness dataset
    with COLUMN_dataset_input:

        # COLUMN title
        st.subheader(":material/upload_file: Dataset Input")
        
        st.markdown("""
            <hr style="margin-top: 5px; margin-bottom: 5px;">
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <p>In this section you will need to upload the dataset (with missing values) for which you want an optimal imputation algorithm <em>“recommendation”</em> to replace the missing data points.</p>
            <p>At this point the only file types that this tool can handle are <code>.csv</code> and <code>.json</code>.</p>
        """, unsafe_allow_html=True)

        # SEPARATOR
        st.markdown('<br>', unsafe_allow_html=True)


        # CONTAINER - Upload Dataset Box
        with st.container():
            
            # Title
            st.markdown("<h5>Upload Dataset</h5>", unsafe_allow_html=True)

            st.markdown("""
                <hr style="margin-top: 5px; margin-bottom: 5px;">
            """, unsafe_allow_html=True)

            st.markdown('<div class="upload-box"> <div class="upload-text"><b>Upload dataset</b><br><small>Dataset file must be of type .csv or .json</small></div></div>', unsafe_allow_html=True)
        
            # SEPARATOR
            st.markdown('<div class="small-space"></div>', unsafe_allow_html=True)

            # File uploader
            uploaded_file = st.file_uploader("", type=["csv", "json"], accept_multiple_files=False, label_visibility="collapsed")


            if not uploaded_file:
                st.markdown('<div class="file-info">No file uploaded yet</div>', unsafe_allow_html=True)

            else:


                try:
                      # Part 1 - Dataset input
                    df = get_file(uploaded_file)



                    # Part 2 - Dataset cleaning
                    uploaded_file_df, uploaded_file_missingness_column = clean_data(df)

                        
                except Exception as e:
                    dataset_warning = True
                    dataset_warning_message = str(e)
                
                else:
                    dataset_warning = False    
            

                if dataset_warning:                
                    st.warning(f"Dataset error {dataset_warning_message}")


        # SEPARATOR
        st.markdown('<br><br><br>', unsafe_allow_html=True)


        # CONTAINER - Dropdown: Missing-ness Mechanism        
        with st.container():

            # Title
            st.markdown("<h5>Select missingness mechansim</h5>", unsafe_allow_html=True)

            st.markdown("""
                <hr style="margin-top: 5px; margin-bottom: 5px;">
            """, unsafe_allow_html=True)

            st.markdown('<div class="dropdown-box"><b>Select missing-ness mechanism</b><br><small>Missingness mechanism must be chosen from one of the options in the drop down box below</small></div>', unsafe_allow_html=True)
            
            # SEPARATOR
            st.markdown('<div class="small-space"></div>', unsafe_allow_html=True)
            
            # Select box
            mechanism = st.selectbox("Missingness mechansim", ["MCAR: (Missing Completely At Random)", "MAR: (Missing At Random)", "MNAR: (Missing Not At Random)"])

            st.text_area("Description:", missingness_mechansims_description_lookup[mechanism.split(":")[0].strip().upper()], height=120)
        

        # SEPARATOR
        st.markdown('<br>', unsafe_allow_html=True)


    # COLUMN - Select the missingness mechanism
    with COLUMN_imputation_algorithm_output:

        # COLUMN title
        st.subheader(":material/output_circle: Imputation Algorithm Output")

        st.markdown("""
            <hr style="margin-top: 5px; margin-bottom: 5px;">
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <p>In this section the optimal imputation algorithm recommendation to replace the missing data points in the uploaded dataset (with missing values) will be displayed along with a description of the imputation algorithm.</p>
        """, unsafe_allow_html=True)
        
        st.markdown('<br>', unsafe_allow_html=True)


        # CONTAINER - Textbox: Display imputation algorithm "recommendation"
        with st.container():
            st.markdown("<h5>Optimal imputation algorithm recommendation</h5>", unsafe_allow_html=True)

            st.markdown("""
                <hr style="margin-top: 5px; margin-bottom: 5px;">
            """, unsafe_allow_html=True)


            if (uploaded_file_df is not None and uploaded_file_missingness_column and mechanism):


                try:

                    # Part 3 - Metafeatures extraction
  
                    # Get statistical metafeatures
                    statistical_metafeatures = get_statistical_metafeatures(df, uploaded_file_missingness_column)

                    # Get missingness metafeaturews
                    missingness_metafeatures = get_missingness_metafeatures(df, uploaded_file_missingness_column, mechanism.split(":")[0].strip().upper())

                    # Get dataset metafeatures
                    metafeatures = [list(statistical_metafeatures.values()) + list(missingness_metafeatures.values())]




                    # Part 4 - Model prediction
                    imputation_algorithm_recommendation = model_recommendation(metafeatures, MODEL_PATH, ENCODED_CLASSESS_PATH)
                
                except Exception as e:
                    pipline_warning = True
                    pipeline_warning_message = str(e)
                    imputation_algorithm_recommendation = "..."
                
                else:
                    pipline_warning = False
            
            else:
                imputation_algorithm_recommendation = "..."


            st.text_area("", imputation_algorithm_recommendation, height=80, label_visibility="collapsed")

            
            if pipline_warning:                
                st.warning(f"Pipeline error {pipeline_warning_message}")

        
        # SEPARATOR
        st.markdown('<br><br><br>', unsafe_allow_html=True)


        # CONTAINER - Textbox: Display imputation algorithm "recommendation" 
        with st.container():
            st.markdown("<h5>Optimal imputation algorithm description</h5>", unsafe_allow_html=True)
            
            st.markdown("""
                <hr style="margin-top: 5px; margin-bottom: 5px;">
            """, unsafe_allow_html=True)


            if imputation_algorithm_recommendation:

                    
                try:
                    imputation_algorithm_recommendation_description = imputation_algorithms_description_lookup[imputation_algorithm_recommendation]

                except Exception as e:
                    imputation_algorithm_recommendation_description = "..."
            
            else:
                imputation_algorithm_recommendation_description = "..."


            st.text_area("", imputation_algorithm_recommendation_description, height=150, label_visibility="collapsed")