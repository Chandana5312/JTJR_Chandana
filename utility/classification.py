import os
import re
import json
import requests


class JobTitleClassification:
    def __init__(self):
        pass

    def classifier(self, prompt):
        headers = {
            "Content-Type": "application/json",
            "api-key": os.getenv("AZURE_OPENAI_API_KEY"),
        }
        data = {
            "model": "gpt-4o",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.01
        }

        response = requests.post(
            f"{os.getenv('AZURE_OPENAI_API_BASE')}/openai/deployments/gpt-4o/chat/completions?api-version=2023-05-15",
            headers=headers,
            json=data,
            timeout=15
        )

        if response.status_code != 200:
            raise Exception(f"Classification API call failed: {response.status_code} - {response.text}")

        response_data = response.json()
        gpt_response = response_data['choices'][0]['message']['content'].strip()

        # Extract the first valid JSON object from the GPT response
        json_match = re.search(r"\{.*?\}", gpt_response, re.DOTALL)
        if not json_match:
            raise ValueError(f"No valid JSON found in GPT response:\n{gpt_response}")

        final_dict = json.loads(json_match.group(0))
        final_dict['input_tokens'] = response_data['usage']['prompt_tokens']
        final_dict['output_tokens'] = response_data['usage']['completion_tokens']
        return final_dict

    def predict(self, job_entry: dict):
        job_title = job_entry['translated_job_title']

        prompt = f"""You are given a job title entered by a user. Your task is to determine whether the given job title is valid or invalid based on the following criteria:

Validation Rules:
A. Valid Job Title:
  - Common abbreviations like CEO, CTO are allowed.
  - Minor typos can be corrected.
  - If the job title contains noise but still clearly includes a valid role, extract that role.
  - Include and preserve seniority, domain, or tech terms where applicable.

B. Invalid Job Title:
  - Pure gibberish, only emojis, only symbols, or random characters.

Use the provided job title as-is. Do not use any alternate fields or LS Title.

Job title: {job_title}

Respond with only a JSON in this format:
{{
  "job title": "<corrected or cleaned job title>",
  "Status": "<Valid or Invalid>"
}}"""

        return self.classifier(prompt)
