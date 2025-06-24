import os
import re
import warnings
import requests
import json
import pandas as pd

import nest_asyncio
from dotenv import load_dotenv
from rapidfuzz import fuzz  

warnings.filterwarnings("ignore")
nest_asyncio.apply()
load_dotenv()

class BestJobRoleFinder():
    def __init__(self, csv_mapping_path="approved_mappings.csv"):
        self.mapping_dict = {}
        if os.path.exists(csv_mapping_path):
            df = pd.read_csv(csv_mapping_path)
            self.mapping_dict = {
                re.sub(r"\s+", " ", str(row['job_title']).strip().lower()): str(row['matched_standard_role']).strip()
                for _, row in df.iterrows()
                if pd.notna(row['job_title']) and pd.notna(row['matched_standard_role'])
            } 
        else:
            print(f"Warning: CSV mapping file {csv_mapping_path} not found.")

    def generate_description(self, job_entry):
        job_title = job_entry['job title']
        combined_text = f"Job Title: {job_title}."

        headers = {
            "Content-Type": "application/json",
            "api-key": os.getenv("AZURE_OPENAI_API_KEY"),
        }

        data = {
            "model": os.getenv("AZURE_OPENAI_GPT_MODEL"),
            "messages": [
                {
                    "role": "user",
                    "content": f"""Generate a list of key responsibilities for the job role, focusing solely on the main tasks and functions. Do not include any information about the company, location, or qualifications. Provide the response in a paragraph format. Be concise and accurate. If the role does not make sense, provide response accordingly saying the word or role does not make sense.\n{combined_text}"""
                }
            ],
            "temperature": 0.01
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

    def get_best_job_role(self, job_entry, top_n_jr_list):
        job_title = job_entry['job title'].strip()
        job_title_lower = job_title.lower()
        job_title_normalized = re.sub(r"\s+", " ", job_title_lower)

        print(f"🔍 Checking approved CSV mappings for: '{job_title_normalized}'")

        # --- Step 1: Exact Match Check ---
        if job_title_normalized in self.mapping_dict:
            matched_role = self.mapping_dict[job_title_normalized]
            print(f"✅ Match found in CSV: '{job_title_normalized}' → '{matched_role}'")
            return {
                "user_job_title": job_title,
                "matched_standard_role": matched_role,
                "confidence_score": 1.0,
                "source": "CSV Mapping"
            }, 0, 0
            
        print(f"❌ No exact match in CSV for: '{job_title_normalized}'")

        # --- Step 2: Fuzzy Match Check ---
        best_match = None
        best_score = 0
        threshold = 90

        for key in self.mapping_dict:
            score = fuzz.token_sort_ratio(job_title_normalized, key)
            if score > best_score:
                best_score = score
                best_match = key

        if best_score >= threshold:
            matched_role = self.mapping_dict[best_match]
            print(f"🤖 Fuzzy match found: '{job_title_normalized}' ≈ '{best_match}' → '{matched_role}' (Score: {best_score})")
            return {
                "user_job_title": job_title,
                "matched_standard_role": matched_role,
                "confidence_score": round(best_score / 100, 2),
                "source": "CSV Mapping (Fuzzy)"
            }, 0, 0

        print("🧠 Proceeding to GPT mapping...")

        # --- Step 3: GPT Mapping fallback ---
        prompt = f"""You are an expert in job title to standard job role mapping. Your task is to accurately map user-entered job titles to the most suitable standard job roles from a predefined list. Your goal is to identify the best match while considering industry norms, job functions, seniority levels and common abbreviations. You must provide a confidence score for the prediction.

# Guidelines:
1. Compare user-provided job titles with the given list of standard job roles.
2. Prioritize exact or closely related matches based on function, industry and level.
3. Must understand abbreviations (e.g., PM → Project Manager).
4. Do not create new job roles outside the provided list.

# Confidence Score (0–1):
- High (0.8–1): Strong match
- Medium (0.5–0.79): Partial match
- Low (<0.5): Weak match, human review suggested

Job Title: {job_title}

Standard Job Roles:
{top_n_jr_list}

Respond only with a JSON in this format:
{{
  "user_job_title": "<copied input>",
  "matched_standard_role": "<best match or 'No suitable match found'>",
  "confidence_score": <float>
}}"""

        headers = {
            "Content-Type": "application/json",
            "api-key": os.getenv("AZURE_OPENAI_API_KEY"),
        }

        data = {
            "model": os.getenv("AZURE_OPENAI_GPT_MODEL"),
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.01
        }

        response = requests.post(
            f"{os.getenv('AZURE_OPENAI_API_BASE')}/openai/deployments/gpt-4o/chat/completions?api-version=2023-05-15",
            headers=headers,
            json=data
        )

        if response.status_code != 200:
            raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")

        gpt_text = response.json()['choices'][0]['message']['content'].strip()

        match = re.search(r"\{[\s\S]*?\}", gpt_text)
        if not match:
            raise ValueError(f"No valid JSON object in GPT response:\n{gpt_text}")

        try:
            ans = json.loads(match.group(0))
        except json.JSONDecodeError as e:
            raise ValueError(f"Malformed JSON in GPT response:\n{gpt_text}\nError: {e}")

        if ans.get("matched_standard_role", "").lower() == "no suitable match found":
            ans["confidence_score"] = 0.0

        input_tokens = response.json()['usage']['prompt_tokens']
        output_tokens = response.json()['usage']['completion_tokens']

        return ans, input_tokens, output_tokens

    def check_string_content(self, text):
        text = str(text)
        modified_text = re.sub(r'[^\w\s]', ' ', text)
        contains_alpha = bool(re.search(r'[A-Za-z]', text))
        contains_digit = bool(re.search(r'\d', text))
        contains_special = bool(re.search(r'[^\w\s]', text))
        only_special_chars = not contains_alpha and not contains_digit and contains_special
        only_numbers = contains_digit and not contains_alpha and not contains_special and re.fullmatch(r'[\d\s]+', text) is not None
        return modified_text, contains_alpha, only_special_chars, only_numbers

    def find_match_metadata(self, top_n, match_std_jr):
        for dict1 in top_n:
            if dict1['job_role'] == match_std_jr:
                return {
                    "job_role": dict1['job_role'],
                    "seniority": dict1.get('seniority'),
                    "marketing_audience": dict1.get('marketing_audience'),
                    "function": dict1.get('function')
                }
        return None
