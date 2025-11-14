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
import uuid
import time
dotenv.load_dotenv()
from utility.agent import MapperAgent




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

# Define upload/results folders (ensure they exist)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok = True)
RESULTS_FOLDER = "results"
os.makedirs(RESULTS_FOLDER, exist_ok = True)

def safe_write_csv(df: pd.DataFrame, target_path: str) -> str:
    """
    Write CSV safely on Windows:
    1) write to a unique temp file
    2) atomically replace target
    3) if target is locked (PermissionError), write to a timestamped fallback file
    Returns the actual path written.
    """
    base_dir = os.path.dirname(target_path)
    base_name = os.path.splitext(os.path.basename(target_path))[0]
    tmp_path = os.path.join(base_dir, f"{base_name}.tmp_{uuid.uuid4().hex}.csv")

    # Ensure folder exists
    os.makedirs(base_dir, exist_ok=True)

    # Write to temp
    df.to_csv(tmp_path, index=False)

    # Try atomic replace; if locked, fallback to a new name
    try:
        os.replace(tmp_path, target_path)
        return target_path
    except PermissionError:
        fallback = os.path.join(base_dir, f"{base_name}_{int(time.time())}.csv")
        os.replace(tmp_path, fallback)
        return fallback

if Certified_flow == "Bulk Mapping":
    # Streamlit UI
    col1.markdown("**Upload CSV or Excel File**")

    uploaded_file = col1.file_uploader("", type = ["csv", "xlsx", "xls"])
    button = col1.button("Upload", type = "primary")


    if button and uploaded_file is not None:
        # reset state so a new upload is always processed
        st.session_state.progress_status = False
        st.session_state.processed_results = []
        st.session_state.df = None
        st.session_state.file_path = None

        # Save the uploaded file (FIX: set path before write)
        file_name = uploaded_file.name
        st.session_state.file_path = os.path.join(UPLOAD_FOLDER, file_name)
        with open(st.session_state.file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        col1.success(f"File saved successfully: {st.session_state.file_path}")

        # Display the content of the file (robust read + sheet fallback)
        df = None
        ext = os.path.splitext(file_name)[1].lower()

        try:
            if ext in (".xlsx", ".xls"):
                # Try to read 'sample_response_data'; fall back to first sheet if absent
                xls = pd.ExcelFile(st.session_state.file_path)
                sheet_to_read = "sample_response_data" if "sample_response_data" in xls.sheet_names else xls.sheet_names[0]
                df = pd.read_excel(xls, sheet_name=sheet_to_read, dtype=str)

            elif ext == ".csv":
                df = pd.read_csv(st.session_state.file_path, dtype=str)
            else:
                st.error(f"Unsupported file type: {ext}. Please upload .xlsx, .xls, or .csv files.")
                st.stop()

        except Exception as e:
            st.error(f"Error reading the file: {e}")
            st.stop()

        if df is not None:
            df = df.fillna("")
            st.session_state.df = df
        else:
            st.error("Uploaded file could not be read (no dataframe).")
            st.stop()

        # Guards before using .columns
        if "df" not in st.session_state or st.session_state.df is None:
            st.error("No data found in the uploaded file.")
            st.stop()

        if st.session_state.df.empty:
            st.error("The uploaded file is empty.")
            st.stop()


        required_columns = {"Lead ID", "jobtitle", "LS Title",
                            "LS Company", "LS Lead Job Functions",
                            "LS Company Industry", "LS Lead Department",
                            "Linkedln Title", "Bing Title", "Country", "Skills"
                            }

        print(">>>>>>>> Embedding", os.getenv("AZURE_OPENAI_EMB_MODEL"))
        df_cols = set([str(c).strip() for c in st.session_state.df.columns])
        missing_columns = [c for c in required_columns if c not in df_cols]

        if missing_columns:
            st.error("The uploaded file is missing the following required columns: " + ", ".join(missing_columns))
            st.stop()
        else:
            st.success("All required columns are present!")


    col1.divider()
    if st.session_state.file_path and st.session_state.progress_status is False:
        # Dummy processing method with loader
        col2.write("Processing rows...")
        progress_bar = col2.progress(0)
        batch_size = 5
        # FIX: use iloc to avoid label-slicing surprises
        jobtitle_batches = [st.session_state.df.iloc[i:i + batch_size]
                            for i in range(0, len(st.session_state.df), batch_size)]
        
        # def _disp_reason(status):
        #     s = str(status or "")
        #     if s.lower().startswith("valid"):
        #         return "Pass"
        #     if s.lower().startswith("invalid -"):
        #         return s  # already includes the reason
        #     if s.lower().startswith("invalid"):
        #         return "Invalid - Not a recognizable job title"
        #     return "Invalid - Unknown"

        # df["Disposition Reason"] = df["Status"].apply(_disp_reason)

        job_role_agent = MapperAgent()
        st.session_state.processed_results = []
        for i, batch in enumerate(jobtitle_batches):
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = {}

                # 1) Submit all jobs in the batch
                for index, row in batch.iterrows():
                    job_entry = {
                        col: str(row[col]) if pd.notna(row[col]) else ''
                        for col in [
                            'Lead ID', 'jobtitle', 'LS Title',
                            'LS Company', 'LS Lead Job Functions',
                            'LS Company Industry', 'LS Lead Department',
                            "Linkedln Title", "Bing Title", "Country", "Skills"
                        ]
                    }
                    futures[executor.submit(job_role_agent.run, job_entry)] = job_entry

                # 2) Collect results (or errors)
                for future in concurrent.futures.as_completed(futures):
                    job_title = futures[future]  # original job_entry dict
                    try:
                        result = future.result()  # success
                        st.session_state.processed_results.append(result)
                        print(f"Job Title: {job_title}, Result: {result}")
                    except Exception as e:
                        print(f"Error processing job title '{job_title}': {e}")
                        # Append a single fallback error row WITH Disposition Reason
                        st.session_state.processed_results.append({
                            "Lead ID": job_title.get("Lead ID", ""),
                            "input_job_title": job_title.get("jobtitle", ""),
                            "detected_language": None,
                            "translated_job_title": None,
                            "Status": f"Error: {str(e)}",
                            "matched_standard_role": None,
                            "seniority": None,
                            "marketing_audience": None,
                            "function": None,
                            "confidence_score": None,
                            "Disposition Reason": f"Error: {str(e)}",
                        })

                progress_bar.progress((i + 1) / len(jobtitle_batches))

        st.session_state.progress_status = True
        print(st.session_state.processed_results)

    if st.session_state.progress_status:
        st.markdown("**Apply Filters:**")
        df = pd.DataFrame(st.session_state.processed_results)

            # --- Build Disposition Reason for every row ---
        def _build_disposition(row):
            # existing value (from error handler etc.)
            existing = row.get("Disposition Reason")
            if isinstance(existing, str) and existing.strip():
                return existing

            status = str(row.get("Status") or "").strip()
            s = status.lower()

            if s.startswith("valid"):
                # Any kind of valid status
                return "Pass"

            if s.startswith("invalid -"):
                # LLM already returned reason after "Invalid -"
                return status

            if s.startswith("invalid"):
                # Generic invalid with no explicit reason
                return "Invalid - Not a recognizable job title"

            if s.startswith("error"):
                # Keep the error message as reason
                return status

            # Fallback for weird or empty statuses
            return ""

        df["Disposition Reason"] = df.apply(_build_disposition, axis=1)


                # --- Ensure Disposition Reason is always populated based on Status ---
        def _disp_reason_from_status(status):
            s = str(status or "")
            sl = s.lower()

            if sl.startswith("valid"):
                return "Pass"

            # Reason already included, e.g. "Invalid - only special characters"
            if sl.startswith("invalid -"):
                return s

            # Generic invalid
            if sl.startswith("invalid"):
                return "Invalid - Not a recognizable job title"

            # Errors from processing
            if sl.startswith("error"):
                return f"Invalid - Error during processing: {s}"

            if not s or s == "nan":
                return "Invalid - Unknown (empty status)"

            # Fallback: return whatever text we have
            return s

        if "Disposition Reason" not in df.columns:
            df["Disposition Reason"] = df["Status"].apply(_disp_reason_from_status)
        else:
            # Fill any missing/empty reasons
            df["Disposition Reason"] = df.apply(
                lambda row: row["Disposition Reason"]
                if str(row["Disposition Reason"]).strip()
                else _disp_reason_from_status(row["Status"]),
                axis=1
            )




        col7,col8,col9,col10 = st.columns([1,1,1,1])
        df = df[['Lead ID','input_job_title',"detected_language",
                 "Status","matched_standard_role","marketing_audience",
                 "function","seniority","confidence_score","Disposition Reason"]]
        df.rename(columns={"Status": "Valid JT"}, inplace=True)
        df.rename(columns={"detected_language": "Language"}, inplace=True)
        df.rename(columns={"matched_standard_role": "Job Role"}, inplace=True)

        df["Valid JT"] = df["Valid JT"].apply(
            lambda x: (
                "Yes" if isinstance(x, str) and x.lower().startswith("valid")
                else "No" if isinstance(x, str) and (x.lower().startswith("invalid") or x.lower().startswith("error"))
                else x
            )
        )



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

        # ---- SAFE WRITES (no UI change) ----
        VALIDATED_FILE_PATH_TARGET  = os.path.join(RESULTS_FOLDER, "validated_results.csv")
        RAW_FILE_PATH_TARGET        = os.path.join(RESULTS_FOLDER, "all_raw_results.csv")

        validated_df = edited_df[edited_df["Certified"].fillna(False)]
        VALIDATED_FILE_PATH = safe_write_csv(validated_df, VALIDATED_FILE_PATH_TARGET)

        raw_df = df_renamed.drop('Certified', axis=1)
        RAW_FILE_PATH = safe_write_csv(raw_df, RAW_FILE_PATH_TARGET)

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
    linkedln_title = col1.text_input('Linkedln Title')
    bing_title = col2.text_input('Bing Title')
    country = col1.text_input('Country')
    skills = col2.text_input('Skills')

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
        job_entry['Linkedln Title'] = linkedln_title
        job_entry['Bing Title'] = bing_title
        job_entry['Country'] = country
        job_entry['Skills'] = skills


        st.session_state.result_dict = job_role_agent.run(job_entry)
    if st.session_state.result_dict : ##single mapping including columns
        col1.write("Preview of the JT-JR Mapping:")
        print("the result is ", st.session_state.result_dict)
        st.session_state.result_df = pd.DataFrame([st.session_state.result_dict])

        def _disp_reason_from_status(status):
            s = str(status or "")
            sl = s.lower()
            if sl.startswith("valid"):
                return "Pass"
            if sl.startswith("invalid -"):
                return s
            if sl.startswith("invalid"):
                return "Invalid - Not a recognizable job title"
            if sl.startswith("error"):
                return f"Invalid - Error during processing: {s}"
            if not s or s == "nan":
                return "Invalid - Unknown (empty status)"
            return s

        st.session_state.result_df["Disposition Reason"] = \
            st.session_state.result_df["Status"].apply(_disp_reason_from_status)
        st.session_state.result_df = st.session_state.result_df[['input_job_title',
                                                                 "detected_language","Status",
                                                                 "matched_standard_role",
                                                                 "marketing_audience",
                                                                 "function","seniority",
                                                                 "confidence_score",
                                                                 "Disposition Reason"]]

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
                        disabled = ["Job Title","Language","Valid JT", ##include new columns for single mapping
                                    "Job Role", "Function", "Seniority", "Marketing Audience","Disposition Reason"],
                                    hide_index = True,width = 1800)
        # col1.dataframe(st.session_state.df.head(1),width = 800,height = 50,hide_index = True)
