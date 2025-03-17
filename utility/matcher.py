import os
import re
import warnings
import requests

import nest_asyncio
from dotenv import load_dotenv
warnings.filterwarnings("ignore")
nest_asyncio.apply()


class BestJobRoleFinder():
    def __init__(self):
        pass

    def generate_description(self, job_entry):
        job_title = job_entry['job title']

        ls_job_title = f"Alternate Title: {job_entry['LS Title']}." if len(job_entry['LS Title']) > 1 else ""
        ls_company = f"\nCompany: {job_entry['LS Company']}." if len(job_entry['LS Company']) > 1 else ""
        ls_job_functions = f"\nJob Functions: {job_entry['LS Job Functions']}." if len(job_entry['LS Job Functions']) > 1 else ""
        ls_company_industry = f"\nCompany Industry: {job_entry['LS Company Industry']}." if len(job_entry['LS Company Industry']) > 1 else ""
        ls_lead_department = f"\nDepartment: {job_entry['LS Lead Department']}." if len(job_entry['LS Lead Department']) > 1 else ""

        combined_text = f"Job Title: {job_title}. {ls_job_title}. {ls_company} {ls_job_functions} {ls_company_industry} {ls_lead_department}"

        ## INCLUDE LS PROMPT
        headers = {
            "Content-Type": "application/json",
            "api-key": os.getenv("AZURE_OPENAI_API_KEY"),
        }
        data = {
            "model": os.getenv("AZURE_OPENAI_GPT_MODEL"),
            "messages": [
                {"role": "user", "content": f"""Generate a list of key responsibilities for the job role, focusing solely on the main tasks and functions. Do not include any information about the company, location, or qualifications. Provide the response in a paragraph format. Be consice and accurate. If the role does not make sense, provide response accordingly saying the word or role does not makes sense. \n{combined_text}"""}
            ],
            "temperature":0.01
        }

        response = requests.post(
            f"{os.getenv('AZURE_OPENAI_API_BASE')}/openai/deployments/gpt-4o/chat/completions?api-version=2023-05-15",
            headers=headers,
            json=data
        )

        response_data = response.json()

        description = response_data['choices'][0]['message']['content'].strip()
        input_tokens = response_data['usage']['prompt_tokens']
        output_tokens = response_data['usage']['completion_tokens']

        return description, input_tokens, output_tokens


    def get_best_job_role(self,job_entry,top_n_jr_list):
        """
        Function to get the best job role for a given job title using Azure OpenAI.

        Parameters:
        df (pd.DataFrame): Input DataFrame containing 'Job Title' and 'Top Job Role' columns.
        job_title_of_interest (str): The job title for which to find the best job role.
        AZURE_OPENAI_API_KEY (str): Your Azure OpenAI API key.
        AZURE_OPENAI_API_BASE (str): Your Azure OpenAI API base URL.

        Returns:
        str: The best job role recommended by Azure OpenAI.
        """


        data = {
            "model": os.getenv("AZURE_OPENAI_GPT_MODEL"),
            "messages": [
                {"role": "user", "content": f"""You are an expert in job title to standard job role mapping. Your task is to accurately map user-entered job titles to the most suitable standard job roles from a predefined list. Your goal is to identify the best match while considering industry norms, job functions, seniority levels and common abbreviations. You must provide a confidence score for the prediction.

                # Guidelines and Instructions:
                1. Compare user-provided job titles with the given list of standard job roles.
                2. Prioritize exact or closely related matches based on function, industry and level.
                3. Must understand and  consider abbreviations & acronyms (e.g., "PM" stands for "Project Manager" or "Product Manager", "IT" stands for "Information Technology") before processing.
                4. "Head" usually means director, sometimes manager.
                5. Ensure job function alignment—focus on the core responsibilities rather than superficial keyword similarity.
                6. If multiple matches exist,  identify the primary role based on job importance.
                7. If the title includes multiple functions (e.g., Product Manager & UX Designer), select the primary role.
                8. If the title is ambiguous, default to the closest general category.
                9. Do not create or suggest any new job roles outside the provided list. If the user provided job title doesn't find relatable match from the standard job roles, return 'No suitable match found.'

                # Assign a Confidence Score (0-1):
                A. High Confidence (0.80-1): Strong match based on job function, industry, and seniority.
                B. Medium Confidence (0.5-0.79): Partial match or ambiguity in the role, no seniority.
                C. Low Confidence (0-0.49): Weak match, requiring human review.

                user provided job title: {job_entry['job title']}
                alternate title : {job_entry['LS Title']}

                standard job roles:
                 {top_n_jr_list}


                Must return the user-entered job title and the best-matching standard job role and a confidence score in following JSON format:
                {{
                "user_job_title": <user provided job title>,
                "matched_standard_role": <most suitable standard job roles>,
                "confidence_score":  <Confidence Score>
                }}
                Only return the JSON output. Do not provide any explanation.
                """}
            ],
            "temperature":0.01
        }
        # print(data)

        headers = {
            "Content-Type": "application/json",
            "api-key": os.getenv("AZURE_OPENAI_API_KEY"),
        }


        response = requests.post(
            f"{os.getenv('AZURE_OPENAI_API_BASE')}/openai/deployments/gpt-4o/chat/completions?api-version=2023-05-15",
            headers=headers,
            json=data
        )
        ans=response.json()['choices'][0]['message']['content'].strip()
        print(ans)
        ans=eval(ans[ans.find("{"):ans.rfind("}")+1])
        input_tokens = response.json()['usage']['prompt_tokens']
        output_tokens = response.json()['usage']['completion_tokens']

        if response.status_code == 200:
            return ans,input_tokens,output_tokens
        else:
            return f"Error: {response.status_code}, {response.text}"


    def check_string_content(self,text):
        text=str(text)
        modified_text = re.sub(r'[^\w\s]', ' ', text)  # Replace special chars with spaces
        contains_alpha = bool(re.search(r'[A-Za-z]', text))  # Check if A-Z exists
        contains_digit = bool(re.search(r'\d', text))  # Check if 0-9 exists
        contains_special = bool(re.search(r'[^\w\s]', text))  # Check if special chars exist
        only_special_chars = not contains_alpha and not contains_digit and contains_special  # Only special characters
        only_numbers = contains_digit and not contains_alpha and not contains_special and re.fullmatch(r'[\d\s]+', text) is not None  # Only numbers with spaces
        # print(" //////////////// only_special_chars", only_special_chars)
        return modified_text, contains_alpha, only_special_chars, only_numbers


    def find_match_metadata(self,top_n,match_std_jr):
        for dict1 in top_n:
            if dict1['job_role']==match_std_jr:
                return {"job_role":dict1['job_role'],
                "seniority":dict1['seniority'],
                "marketing_audience":dict1['marketing_audience'],
                "function":dict1['function']}
        return None
