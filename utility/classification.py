import requests
import os



class JobTitleClassification:
    
    def __init__(self):
        pass

    def classifier(self,prompt):
        """Generates a list of key responsibilities for the role, focusing solely on the main tasks and functions.
        Does not include any information about the company, location, or qualifications.
        Provide the response in a paragraph format.Be consice and accurate.
        If the role does not make sense, provide response accordingly saying the word or role does not makes sense"""

        headers = {
            "Content-Type": "application/json",
            "api-key": os.getenv("AZURE_OPENAI_API_KEY"),
        }
        data = {
            "model": "gpt-4o",  
            "messages": [
                {"role": "user", "content":prompt }
            ],
            "temperature":0.01
            
        }
        response = requests.post(
            f"""{os.getenv("AZURE_OPENAI_API_BASE")}/openai/deployments/gpt-4o/chat/completions?api-version=2023-05-15""",
            headers=headers,
            json=data
        )
        response_data = response.json()
        # description = response_data['choices'][0]['message']['content'].strip()
        input_tokens = response_data['usage']['prompt_tokens']  
        output_tokens = response_data['usage']['completion_tokens']  
        return response_data,input_tokens,output_tokens
    
    def predict(self,job_entry:dict):
        ### NEED TO PASS DICT HERE
        # job_title = job_entry['input_job_title']
        job_title = job_entry['translated_job_title']
        ls_title = job_entry['LS Title']
 
        prompt=f"""You are given a job title entered by a user. Your task is to determine whether the given job title is valid or invalid based on the following criteria:
        
        Validation Rules:
        A. Valid Job Title:
            1. The job title should be a recognizable professional role or closely resemble one (e.g., "Software Engineer", "Data Scientist", "Marketing Manager").
            2. It may include common industry terms and roles.
            3. Standard abbreviations for job titles are acceptable (e.g., "CEO", "CTO").
            4. If the job title contains minor spelling mistakes (e.g., "Sfotware Engneer" → "Software Engineer"), correct it to the closest valid job title.
            5. If a job title includes unnecessary words that are not part of a professional title but still contains a valid job title (e.g., "IT and understand Data Scientist" → "Data Scientist"), extract the relevant role while preserving domain, technology and seniority if applicable.
            6. If a job title includes words unrelated to a professional role but still contains a valid job title, extract the relevant role while preserving:
                a) Seniority (e.g., "Assistant", "Senior", "Lead"). b) Domain or Industry (e.g., "Cyber Security", "Logistics", "Finance"). c) Technology or Specialization (e.g., "AI", "Cloud", "Data Science").
            7. While extracting the above information do not force remove the corresponding job title until the input job title is completely or closely clear. For example. Student of data science >> to not mapped as valid.

        B. Invalid Job Title:
            1. The input consists only of emojis (e.g., "🔥", "💼").
            2. Includes gibberish, random characters or meaningless words (e.g., "asdfgh", "xxxyyy").
            3. The title includes excessive punctuation or symbols, making it unidentifiable (e.g., "@@", "!!!").
            4. Includes explicit, offensive, or unrelated terms.
            5. The job title not recognizable professional role or not closely resemble to the professional role.
            6. Do not remove anything if the job title is invalid. Return the input job title as is.
            
        User given job title: {job_title}
        Alternate title : {ls_title}
        
        If any of the titles are valid and similar, give the most suitable job title.
        If either the user given job title or the alternate title is valid, use only the valid one.
        If the user given job title and the alternate tile both are valid but contradicts each other, prioretise the aternate title.

        The final output should be in JSON format, following this structure:
        {{
          "job title": "<corrected or the most suitable job title>",
          "Status": "<Valid or Invalid>"}}"""
        response, input_tokens, output_tokens = self.classifier(prompt)
        ans = response['choices'][0]['message']['content'].strip()
        final_dict = ans[ans.find("{"):ans.find("}")+1]
        final_dict = eval(final_dict)
        final_dict['input_tokens'] = input_tokens
        final_dict['output_tokens'] = output_tokens
        return final_dict