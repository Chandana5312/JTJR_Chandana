import streamlit as st
import os
import pandas as pd
import time
from utility.classification import JobTitleClassification
from utility.search import SearchAgent
from utility.translator import AzureTranslator
from utility.matcher import BestJobRoleFinder


class MapperAgent():
    def __init__(self):
        self.translator = AzureTranslator()
        self.classifier = JobTitleClassification()
        self.search_agent = SearchAgent()
        self.job_role_finder = BestJobRoleFinder()

    def run(self,job_entry: dict):
        # Final Output dict
        job_title_dict = {}
        job_title = job_entry['jobtitle']
        job_title_dict['Lead ID'] = job_entry['Lead ID']
        print("-------------------------------Input----------------------------------------------------")
        job_title_dict["input_job_title"] = job_title
        print("Input job Title :: ",job_title)
        print()
        print("-------------------------------Step 1----------------------------------------------------")
        print("Step 1 :: Translation & Language Identification Started...")
        start_time_jtt = time.time()
        jt_lang, jobtitle_tr = self.translator.detect_and_translate(job_title)

        end_time_jtt = time.time()
        total_time_jtt = end_time_jtt - start_time_jtt

        job_title_dict['detected_language'] = jt_lang
        job_title_dict['translated_job_title'] = jobtitle_tr
        job_title_dict['jt_trans_time'] = total_time_jtt

        print("Detected Language ::",jt_lang)
        print("Translated Job Title ::",jobtitle_tr)
        print("Step 1 :: Translation & Language Identification Completed...")
        print()

        job_title,contains_alpha,only_special_chars,only_numbers = self.job_role_finder.check_string_content(jobtitle_tr)

        if contains_alpha:
            print("-------------------------------Step 2 A----------------------------------------------------")
            print("Step 2 A:: Job Title Classification Started...")
            print("Step 2 A::",job_title)
            print("Step 2 A::",job_title_dict)
            if pd.notna(job_entry['LS Title']):
                ls_title_lang, ls_title_tr = self.translator.detect_and_translate(job_entry['LS Title'])
                job_title_dict['LS Title'] = ls_title_tr
            else :
                job_title_dict['LS Title'] = ''
            # job title
            start_time_jtc = time.time()

            class_dict = self.classifier.predict(job_title_dict)
            class_dict['Lead ID'] = job_title_dict['Lead ID']
            end_time_jtc = time.time()
            total_time_jtc = end_time_jtc-start_time_jtc

            job_title_dict["Status"] = class_dict['Status']
            job_title_dict["input_tokens_jtc"] = class_dict['input_tokens']
            job_title_dict["output_tokens_jtc"] = class_dict['output_tokens']
            job_title_dict['jt_class_time'] = total_time_jtc
            print("Class Dict Status",class_dict['Status'])
            print("Step 2 A:: Job Title Classification Completed...")
            print("Step 2 A:: class_dict", class_dict)
            print("Step 2 A:: job_title_dict", job_title_dict)
            print()

            if class_dict['Status'].lower() == "valid":
                print("-------------------------------Step 3 A----------------------------------------------------")
                print("Step 3 A:: Job Title Description Generate Started...")

                try:
                    print("Step 3A::", job_entry)
                    ls_company_lang, ls_company_tr = self.translator.detect_and_translate(job_entry['LS Company']) if pd.notna(job_entry['LS Company']) else ''
                    ls_job_functions_lang, ls_job_functions_tr = self.translator.detect_and_translate(job_entry['LS Lead Job Functions']) if pd.notna(job_entry['LS Lead Job Functions']) else ''
                    ls_company_industry_lang, ls_company_industry_tr = self.translator.detect_and_translate(job_entry['LS Company Industry']) if pd.notna(job_entry['LS Company Industry']) else ''
                    ls_lead_department_lang, ls_lead_department_tr = self.translator.detect_and_translate(job_entry['LS Lead Department']) if pd.notna(job_entry['LS Lead Department']) else ''
                    print("Step 3 B:: Translations completed...", )

                except Exception as e:
                    print(f"Failed due to {e}")


                class_dict['LS Title'] = ls_title_tr
                class_dict['LS Company'] = ls_company_tr
                class_dict['LS Job Functions'] = ls_job_functions_tr
                class_dict['LS Company Industry'] = ls_company_industry_tr
                class_dict['LS Lead Department'] = ls_lead_department_tr

                print("Class_dict: ", class_dict)

                start_time_jtd = time.time()
                description, input_tokens1, output_tokens1 = self.job_role_finder.generate_description(class_dict)
                end_time_jtd = time.time()
                total_time_jtd = end_time_jtd-start_time_jtd
                print(description)
                # input_emd_tokens = count_tokens(description)
                job_title_dict["job_title_desc"] = description
                job_title_dict["input_tokens_jtd"] = input_tokens1
                job_title_dict["output_tokens_jtd"] = output_tokens1
                job_title_dict['jt_desc_time'] = total_time_jtd

                # print("input tokens",input_tokens1)
                # print("output tokens",output_tokens1)
                print("Step 3 A:: Job Title Description Generate Completed...")
                print()

                print("-------------------------------Step 4----------------------------------------------------")
                print("Step 4 :: Azure AI Search...")
                start_time_jts = time.time()
                top_5_job_role,input_emd_tokens = self.search_agent.search(description,has_text = True,has_vector = True,use_semantic_captions = False,top = 5)
                end_time_jts = time.time()
                total_time_jts = end_time_jts-start_time_jts

                job_title_dict["emd_tokens_desc"] = input_emd_tokens
                job_title_dict['jt_Search_time'] = total_time_jts
                job_title_dict["TOP N"] = top_5_job_role
                top_n_job_role = [doc['job_role'] for doc in top_5_job_role]
                # print(top_n_job_role)

                print("-------------------------------Step 5----------------------------------------------------")
                print("Step 5 :: Best Job Role...")
                start_time_jtbm = time.time()
                final_match_dict, input_tokens2, output_tokens2 = self.job_role_finder.get_best_job_role(class_dict,top_n_job_role)
                end_time_jtbm = time.time()
                total_time_jtbm = end_time_jtbm-start_time_jtbm
                job_title_dict['jt_best_match_time'] = total_time_jtbm
                job_title_dict["input_tokens_jtbm"] = input_tokens2
                job_title_dict["output_tokens_jtbm"] = output_tokens2
                print(f"{final_match_dict['matched_standard_role']},{final_match_dict['confidence_score']}")
                job_title_dict["matched_standard_role"] = final_match_dict['matched_standard_role']
                match_metadata = self.job_role_finder.find_match_metadata(top_5_job_role,final_match_dict['matched_standard_role'])
                if match_metadata != None:
                    if "seniority" in match_metadata.keys():
                        job_title_dict["seniority"] = match_metadata['seniority']
                    if "marketing_audience" in match_metadata.keys():
                        job_title_dict["marketing_audience"] = match_metadata['marketing_audience']
                    if "function" in match_metadata.keys():
                        job_title_dict["function"] = match_metadata['function']
                else:
                    job_title_dict["seniority"] = None
                    job_title_dict["marketing_audience"] = None
                    job_title_dict["function"] = None

                for i in range(5):
                    job_title_dict[f"job_match_{i}"] = top_n_job_role[i]

                job_title_dict["confidence_score"] = final_match_dict['confidence_score']

                return job_title_dict
            else:
                print("-------------------------------Step 3 B----------------------------------------------------")
                print("Step 3 B :: Invalid Job Title...")
                job_title_dict["TOP N"] = None
                job_title_dict["job_title_desc"] = None
                job_title_dict["emd_tokens_desc"] = 0
                job_title_dict["input_tokens_jtd"] = 0
                job_title_dict["output_tokens_jtd"] = 0
                job_title_dict["matched_standard_role"] = None
                job_title_dict["seniority"] = None
                job_title_dict["marketing_audience"] = None
                job_title_dict["function"] = None
                job_title_dict["confidence_score"] = None
                job_title_dict["input_tokens_jtbm"] = 0
                job_title_dict["output_tokens_jtbm"] = 0
                job_title_dict['jt_desc_time'] = 0
                job_title_dict['jt_Search_time'] = 0
                job_title_dict['jt_best_match_time'] = 0
                return job_title_dict
        else:
            print("-------------------------------Step 2 B ----------------------------------------------------")
            print("Step 2 B :: Invalid - only special characters")
            job_title_dict['Status'] = 'Invalid - only special characters'
            job_title_dict['TOP N'] = None
            job_title_dict['job_title_desc'] = None
            job_title_dict["emd_tokens_desc"] = 0
            job_title_dict['matched_standard_role'] = None
            job_title_dict["seniority"] = None
            job_title_dict["marketing_audience"] = None
            job_title_dict["function"] = None
            job_title_dict['confidence_score'] = None
            job_title_dict["input_tokens_jtc"] = 0
            job_title_dict["output_tokens_jtc"] = 0
            job_title_dict["input_tokens_jtd"] = 0
            job_title_dict["output_tokens_jtd"] = 0
            job_title_dict["input_tokens_jtbm"] = 0
            job_title_dict["output_tokens_jtbm"] = 0
            job_title_dict['jt_class_time'] = 0
            job_title_dict['jt_desc_time'] = 0
            job_title_dict['jt_Search_time'] = 0
            job_title_dict['jt_best_match_time'] = 0
            return job_title_dict
