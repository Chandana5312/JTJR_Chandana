"""
app.py

This module is the main entry point for the application. It initializes the Streamlit UI,
handles file processing, and manages session state.
"""

import os
import concurrent.futures
import base64
import dotenv
import pandas as pd
import streamlit as st

from utility.agent import MapperAgent

dotenv.load_dotenv()


# https://acis.affineanalytics.co.in/assets/images/logo_small.png
# <img src = "#" alt = "logo" width = "70" height = "60">
if 'result_dict' not in st.session_state:
    st.session_state.result_dict = {}
st.set_page_config(page_title = "JTJR", layout = "wide")
# st.logo("ms_logo.png", size = 'medium')  # Ensure "ms_logo.png" is in the same directory

with open("ms_logo.png", "rb") as ms_logo:
    ms_logo_b64 = base64.b64encode(ms_logo.read()).decode("utf-8")


    # <div style = 'text-align: center; margin-top:-50px; margin-bottom: 5px;margin-left:0px;'>
st.markdown(f"""
            <div style = "display: flex; align-items: center; justify-content: center; gap: 10px; text-align: center;">
    <img src = "data:image/png;base64,{ms_logo_b64}" style = "width: 100px; height:auto;">
    <h2 style = 'font-size: 50px; font-family: 'Arial', Geneva, Verdana, sans-serif;;
                    letter-spacing: 0px; text-decoration: none; text-align: center;'>
    <span style = 'background: linear-gradient(45deg,  #0044cc, #6a85b6);
                            -webkit-background-clip: text;
                            -webkit-text-fill-color: transparent;
                            text-shadow: none;'>
                    Title to Role Standardization
    </span>
    <span style = 'font-size: 60%;'>
    <sup style = 'position: relative; top: 5px; color:white ;'></sup>
    </span>
    </h2>
    </div>
    """, unsafe_allow_html = True)

Certified_flow = st.radio("**Select the Flow:**",
                          ["Single Mapping",  "Bulk Mapping"],horizontal = True)
st.write(" ")
st.write(" ")

col1,col2 = st.columns([3,3])

if "file_path" not in st.session_state:
    st.session_state.file_path = None

if "df" not in st.session_state:
    st.session_state.df = None

if "progress_status" not in st.session_state:
    st.session_state.progress_status = False

# Define upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok = True)


if Certified_flow == "Bulk Mapping":
    # Streamlit UI
    col1.markdown("**Upload CSV or Excel File**")

    uploaded_file = col1.file_uploader("", type = ["csv", "xlsx", "xls"])
    button = col1.button("Upload", type = "primary")


    if button and uploaded_file is not None and st.session_state.file_path is None:
        file_name = uploaded_file.name
        st.session_state.file_path = os.path.join(UPLOAD_FOLDER, file_name)

        # Save the uploaded file
        with open(st.session_state.file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        col1.success(f"File saved successfully: {st.session_state.file_path}")

        # Display the content of the file
        try:
            if file_name.endswith(".csv"):
                st.session_state.df = pd.read_csv(st.session_state.file_path)
            else:
                st.session_state.df = pd.read_excel(st.session_state.file_path,
                                                    sheet_name = "sample_response_data")

            # st.write("Preview of the uploaded file:")
            # st.dataframe(st.session_state.df.head())
        except Exception as e:
            st.error(f"Error reading the file: {e}")

        required_columns = {"Lead ID", "jobtitle", "LS Title",
                            "LS Company", "LS Lead Job Functions",
                            "LS Company Industry", "LS Lead Department"
                            }

        print(">>>>>>>> Embedding", os.getenv("AZURE_OPENAI_EMB_MODEL"))
        missing_columns = required_columns - set(st.session_state.df.columns)

        if missing_columns:
            st.error(f"Missing columns: {', '.join(missing_columns)}")
        else:
            st.success("All required columns are present!")


    col1.divider()
    if st.session_state.file_path and st.session_state.progress_status is False:
        # Dummy processing method with loader
        col2.write("Processing rows...")
        progress_bar = col2.progress(0)
        batch_size = 5
        jobtitle_batches = [st.session_state.df[i:i + batch_size]
                            for i in range(0, len(st.session_state.df), batch_size)]

        # for batch in jobtitle_batches:

        job_role_agent = MapperAgent()
        st.session_state.processed_results = []
        for i, batch in enumerate(jobtitle_batches):
            with concurrent.futures.ThreadPoolExecutor(max_workers = 5) as executor:
                futures = {}
                for index, row in batch.iterrows():
                    job_entry = ({
                        col: str(row[col]) if pd.notna(row[col]) else ''
                        for col in ['Lead ID', 'jobtitle', 'LS Title',
                                    'LS Company', 'LS Lead Job Functions',
                                    'LS Company Industry', 'LS Lead Department']
                    })

                    futures[executor.submit(job_role_agent.run, job_entry)] = job_entry

                for future in concurrent.futures.as_completed(futures):
                    job_title = futures[future]
                    try:
                        result = future.result()  # Get result of the job
                        st.session_state.processed_results.append(future.result())
                        print(f"Job Title: {job_title}, Result: {result}")
                    except Exception as e:
                        print(f"Error processing job title '{job_title}': {e}")

                progress_bar.progress((i + 1) / len(jobtitle_batches)) # text = row.input_job_title
            st.session_state.progress_status = True
            print(st.session_state.processed_results)

    if st.session_state.progress_status:
        st.markdown("**Apply Filters:**")
        df = pd.DataFrame(st.session_state.processed_results)



        col7,col8,col9,col10 = st.columns([1,1,1,1])
        df = df[['Lead ID','input_job_title',"detected_language",
                 "Status","matched_standard_role","marketing_audience",
                 "function","seniority","confidence_score"]]
        df.rename(columns={"Status": "Valid JT"}, inplace=True)
        df.rename(columns={"detected_language": "Language"}, inplace=True)
        df.rename(columns={"matched_standard_role": "Job Role"}, inplace=True)


        # Replace values
        # df["Valid JT"] = df["Valid JT"].replace({"Valid": "Yes", "Invalid": "No"})
        df["Valid JT"] = df["Valid JT"].apply(lambda x: "No"
                                              if "Invalid" in x else "Yes" if "Valid" in x else x)

        detected_language_list = df["Language"].dropna().unique()
        seniority_list = df["seniority"].dropna().unique()
        valid_jobs = df["Valid JT"].dropna().unique()
        matched_std_role_list = df["Job Role"].dropna().unique()
        marketing_audience_list = df["marketing_audience"].dropna().unique()
        function_list = df["function"].dropna().unique()

        select_detected_lang = col7.multiselect("Language",detected_language_list)
        select_job_title_validity = col7.multiselect("Valid JT",valid_jobs)

        select_seniority = col8.multiselect("Seniority",seniority_list)
        select_matched_std_role_list = col8.multiselect("Job Role", matched_std_role_list)

        select_marketing_audience = col9.multiselect("Marketing Audience", marketing_audience_list)
        select_function = col9.multiselect("Function", function_list)

        confidence_threshold = col10.slider("Confidence Score",
                                            min_value = 0.0,max_value = 1.0,step = 0.1)

        st.divider()

        df['Certified'] = False

        # print(confidence_threshold)
        if len(select_detected_lang)>0:
            df = df[df["Language"].isin(select_detected_lang)]
        if len(select_seniority)>0:
            df = df[df["seniority"].isin(select_seniority)]
        if len(select_job_title_validity) > 0:
            df = df[df["Valid JT"].isin(select_job_title_validity)]
        if len(select_function) > 0:
            df = df[df["function"].isin(select_function)]
        if len(select_marketing_audience) > 0:
            df = df[df["marketing_audience"].isin(select_marketing_audience)]
        if len(select_matched_std_role_list) > 0:
            df = df[df["Job Role"].isin(select_matched_std_role_list)]


        if confidence_threshold>0:
            print(f"-----------------{confidence_threshold}----------------------------------")
            df = df[df["confidence_score"] >= confidence_threshold]
        st.write()
        st.write("**Preview of the JT-JR Mapping:**")

        col5,col6,col_ = st.columns([8,1,1])
        select_button = col6.button("Certify All")
        unselect_button = col_.button("Uncertify All")

        if select_button:
            df['Certified'] = True
        if unselect_button:
            df['Certified'] = False

        df_renamed = df.copy()

        # Rename multiple columns at once
        df_renamed.rename(columns={
            "function": "Function",
            "marketing_audience": "Marketing Audience",
            "matched_standard_role": "Job Role",
            "seniority" : "Seniority",
            "input_job_title" : "Job Title",
            "confidence_score": "Confidence Score"
        }, inplace=True)
        # st.dataframe(df,height = 200,hide_index = True)
        edited_df = st.data_editor(df_renamed, key = "table_editor",
                                   num_rows = "dynamic",
                                   disabled = ["Job Title","Lead ID",
                                                "Job Role", "Seniority",
                                                "Confidence Score", "Language",
                                                "Valid JT", "Marketing Audience", "Function"],
                                    hide_index = True,width = 1800)

        VALIDATED_FILE_PATH  = "results/validated_results.csv"
        # if not os.path.exists(VALIDATED_FILE_PATH ):
        validated_df = edited_df[edited_df["Certified"].fillna(False)]
        validated_df.to_csv(VALIDATED_FILE_PATH , index=False)
        RAW_FILE_PATH  = "results/all_raw_results.csv"
        # if not os.path.exists(RAW_FILE_PATH ):
        df_renamed.drop('Certified', axis=1).to_csv(RAW_FILE_PATH , index=False)

        col11, col12, col13 = st.columns([1,1,3])
        with open(RAW_FILE_PATH , "rb") as file:
            btn = col11.download_button(label = "Download All Mapping",
                                        data = file,file_name = "all_raw_results.csv")

        with open(VALIDATED_FILE_PATH , "rb") as file:
            btn = col12.download_button(label = "Download Certified Mapping",
                                        data = file,file_name = "validated_mapping_results.csv")
else:

    input_job_title = col1.text_input("Job Title")
    ls_title = col1.text_input('LS Title')
    ls_company = col1.text_input( 'LS Company')
    ls_lead_job = col2.text_input('LS Lead Job Functions')
    ls_comapny_industry = col2.text_input('LS Company Industry')
    ls_lead_dept = col2.text_input('LS Lead Department')
    button1 = col1.button("Submit", type = "primary")
    col1.divider()
    job_entry  = {}
    job_role_agent = MapperAgent()
    if button1:
        job_entry['jobtitle'] = input_job_title
        job_entry['LS Title'] =  ls_title
        job_entry['LS Company'] = ls_company
        job_entry['LS Lead Job Functions']  = ls_lead_job
        job_entry['LS Company Industry']  = ls_comapny_industry
        job_entry['LS Lead Department'] = ls_lead_dept
        job_entry['Lead ID'] = 0

        st.session_state.result_dict = job_role_agent.run(job_entry)
    if st.session_state.result_dict :
        col1.write("Preview of the JT-JR Mapping:")
        print("the result is ", st.session_state.result_dict)
        st.session_state.result_df = pd.DataFrame([st.session_state.result_dict])
        st.session_state.result_df = st.session_state.result_df[['input_job_title',
                                                                 "detected_language","Status",
                                                                 "matched_standard_role",
                                                                 "marketing_audience",
                                                                 "function","seniority",
                                                                 "confidence_score"]]

        df_renamed = st.session_state.result_df.copy()

        # Rename multiple columns at once
        df_renamed.rename(columns={
            "function": "Function",
            "marketing_audience": "Marketing Audience",
            "detected_language" : "Language",
            "matched_standard_role": "Job Role",
            "Status" : "Valid JT",
            "seniority" : "Seniority",
            "input_job_title" : "Job Title",
            "confidence_score": "Confidence Score"
        }, inplace=True)

        print("the length is ",len(st.session_state.result_df))
        st.data_editor(df_renamed, key = "table_editor", num_rows = "dynamic",
                        disabled = ["Job Title","Language","Valid JT",
                                    "Job Role", "Function", "Seniority", "Marketing Audience"],
                                    hide_index = True,width = 1800)
        # col1.dataframe(st.session_state.df.head(1),width = 800,height = 50,hide_index = True)
