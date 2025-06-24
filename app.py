import os
import concurrent.futures
import base64
import dotenv
import pandas as pd
import streamlit as st
 
dotenv.load_dotenv()
from utility.agent import MapperAgent
 
if 'result_dict' not in st.session_state:
    st.session_state.result_dict = {}
 
st.set_page_config(page_title="JTJR", layout="wide")
 
with open("ms_logo.png", "rb") as ms_logo:
    ms_logo_b64 = base64.b64encode(ms_logo.read()).decode("utf-8")
 
st.markdown(f"""
    <div style = "display: flex; align-items: center; justify-content: center; gap: 10px; text-align: center;">
    <img src = "data:image/png;base64,{ms_logo_b64}" style = "width: 100px; height:auto;">
    <h2 style = 'font-size: 50px; font-family: Arial, Geneva, Verdana, sans-serif;'>
    <span style = 'background: linear-gradient(45deg,  #0044cc, #6a85b6);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                text-shadow: none;'>
                Title to Role Standardization
    </span>
    </h2>
    </div>
""", unsafe_allow_html=True)
 
Certified_flow = st.radio("**Select the Flow:**", ["Single Mapping",  "Bulk Mapping"], horizontal=True)
st.write("\n\n")
col1, col2 = st.columns([3, 3])
 
if "file_path" not in st.session_state:
    st.session_state.file_path = None
if "df" not in st.session_state:
    st.session_state.df = None
if "progress_status" not in st.session_state:
    st.session_state.progress_status = False
 
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
 
if Certified_flow == "Bulk Mapping":
    col1.markdown("**Upload CSV or Excel File**")
    uploaded_file = col1.file_uploader("", type=["csv", "xlsx", "xls"])
    button = col1.button("Upload", type="primary")
 
    if button and uploaded_file and st.session_state.file_path is None:
        file_name = uploaded_file.name
        st.session_state.file_path = os.path.join(UPLOAD_FOLDER, file_name)
        with open(st.session_state.file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        col1.success(f"File saved successfully: {st.session_state.file_path}")
 
        try:
            if file_name.endswith(".csv"):
                st.session_state.df = pd.read_csv(st.session_state.file_path)
            else:
                st.session_state.df = pd.read_excel(st.session_state.file_path, sheet_name="sample_response_data")
        except Exception as e:
            st.error(f"Error reading the file: {e}")
 
        required_columns = {"Lead ID", "jobtitle"}
        missing_columns = required_columns - set(st.session_state.df.columns)
 
        if missing_columns:
            st.error(f"Missing columns: {', '.join(missing_columns)}")
        else:
            st.success("All required columns are present!")
 
    col1.divider()
    if st.session_state.file_path and not st.session_state.progress_status:
        col2.write("Processing rows...")
        progress_bar = col2.progress(0)
        batch_size = 5
        jobtitle_batches = [st.session_state.df[i:i + batch_size] for i in range(0, len(st.session_state.df), batch_size)]
 
        job_role_agent = MapperAgent()
        st.session_state.processed_results = []
 
        for i, batch in enumerate(jobtitle_batches):
            job_entries = []
            for index, row in batch.iterrows():
                job_entry = {
                    col: str(row[col]) if pd.notna(row[col]) else ''
                    for col in ['Lead ID', 'jobtitle']
                }
                job_entries.append(job_entry)
 
            def process_entry(job_entry):
                try:
                    return job_role_agent.run(job_entry)
                except Exception as e:
                    st.error(f"❌ Error processing job title '{job_entry['jobtitle']}': {e}")
                    return {
                        "Lead ID": job_entry['Lead ID'],
                        "input_job_title": job_entry['jobtitle'],
                        "Status": "Invalid",
                        "matched_standard_role": None,
                        "confidence_score": 0
                    }
 
            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                results = list(executor.map(process_entry, job_entries))
                st.session_state.processed_results.extend(results)
 
            progress_bar.progress((i + 1) / len(jobtitle_batches))
 
        st.session_state.progress_status = True
 
    if st.session_state.progress_status:
        st.markdown("**Apply Filters:**")
        df = pd.DataFrame(st.session_state.processed_results)
        col7, col8, col9, col10 = st.columns([1, 1, 1, 1])
 
        df = df[['Lead ID','input_job_title','detected_language','Status','matched_standard_role',
                 'marketing_audience','function','seniority','confidence_score']]
 
        df.rename(columns={"Status": "Valid JT", "detected_language": "Language",
                           "matched_standard_role": "Job Role"}, inplace=True)
 
        df["Valid JT"] = df["Valid JT"].apply(lambda x: "No" if "Invalid" in x else "Yes" if "Valid" in x else x)
 
        filters = {
            "Language": col7.multiselect("Language", df["Language"].dropna().unique()),
            "Valid JT": col7.multiselect("Valid JT", df["Valid JT"].dropna().unique()),
            "Seniority": col8.multiselect("Seniority", df["seniority"].dropna().unique()),
            "Job Role": col8.multiselect("Job Role", df["Job Role"].dropna().unique()),
            "Marketing Audience": col9.multiselect("Marketing Audience", df["marketing_audience"].dropna().unique()),
            "Function": col9.multiselect("Function", df["function"].dropna().unique()),
        }
 
        confidence_threshold = col10.slider("Confidence Score", 0.0, 1.0, 0.1)
        df['Certified'] = False
 
        for col, values in filters.items():
            if values:
                df = df[df[col].isin(values)]
 
        if confidence_threshold > 0:
            df = df[df["confidence_score"] >= confidence_threshold]
 
        st.write("**Preview of the JT-JR Mapping:**")
        col5, col6, col_ = st.columns([8, 1, 1])
        if col6.button("Certify All"):
            df['Certified'] = True
        if col_.button("Uncertify All"):
            df['Certified'] = False
 
        df_renamed = df.rename(columns={
            "function": "Function", "marketing_audience": "Marketing Audience",
            "seniority": "Seniority", "input_job_title": "Job Title",
            "confidence_score": "Confidence Score"
        })
 
        edited_df = st.data_editor(df_renamed, key="table_editor", num_rows="dynamic",
                                   disabled=["Job Title", "Lead ID", "Job Role", "Seniority",
                                             "Confidence Score", "Language", "Valid JT",
                                             "Marketing Audience", "Function"],
                                   hide_index=True, width=1800)
 
        VALIDATED_FILE_PATH = "results/validated_results.csv"
        validated_df = edited_df[edited_df["Certified"].fillna(False)]
        validated_df.to_csv(VALIDATED_FILE_PATH, index=False)
 
        RAW_FILE_PATH = "results/all_raw_results.csv"
        df_renamed.drop('Certified', axis=1).to_csv(RAW_FILE_PATH, index=False)
 
        col11, col12, _ = st.columns([1, 1, 3])
        with open(RAW_FILE_PATH, "rb") as file:
            col11.download_button("Download All Mapping", file.read(), "all_raw_results.csv")
        with open(VALIDATED_FILE_PATH, "rb") as file:
            col12.download_button("Download Certified Mapping", file.read(), "validated_mapping_results.csv")
 
else:
    input_job_title = col1.text_input("Job Title")
    ls_title = col1.text_input('LS Title')
    ls_company = col1.text_input('LS Company')
    ls_lead_job = col2.text_input('LS Lead Job Functions')
    ls_company_industry = col2.text_input('LS Company Industry')
    ls_lead_dept = col2.text_input('LS Lead Department')
    button1 = col1.button("Submit", type="primary")
    col1.divider()
    job_entry = {}
    job_role_agent = MapperAgent()
    if button1:
        job_entry = {
            'jobtitle': input_job_title,
            'LS Title': ls_title,
            'LS Company': ls_company,
            'LS Lead Job Functions': ls_lead_job,
            'LS Company Industry': ls_company_industry,
            'LS Lead Department': ls_lead_dept,
            'Lead ID': 0
        }
        st.session_state.result_dict = job_role_agent.run(job_entry)
 
    if st.session_state.result_dict:
        col1.write("Preview of the JT-JR Mapping:")
        st.session_state.result_df = pd.DataFrame([st.session_state.result_dict])
        df_renamed = st.session_state.result_df.rename(columns={
            "function": "Function", "marketing_audience": "Marketing Audience",
            "detected_language": "Language", "matched_standard_role": "Job Role",
            "Status": "Valid JT", "seniority": "Seniority",
            "input_job_title": "Job Title", "confidence_score": "Confidence Score"
        })
        st.data_editor(df_renamed, key="table_editor", num_rows="dynamic",
                       disabled=["Job Title", "Language", "Valid JT", "Job Role",
                                 "Function", "Seniority", "Marketing Audience"],
                       hide_index=True, width=1800)