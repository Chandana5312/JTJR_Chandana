
import requests
import json
import os



class AzureTranslator:
    def __init__(self):
        """
        Initialize the AzureTranslator class with API credentials.
        :param subscription_key: Azure Translator API subscription key.
        :param endpoint: Azure Translator API endpoint.
        :param region: Azure service region.
        """
        self.subscription_key = os.getenv("AZURE_TRANSLATOR_KEY")
        self.endpoint = os.getenv("AZURE_TRANSLATOR_ENDPOINT")
        self.region = os.getenv("AZURE_TRANSLATOR_REGION")
        self.headers = {
            'Ocp-Apim-Subscription-Key': self.subscription_key,
            'Ocp-Apim-Subscription-Region': self.region,
            'Content-Type': 'application/json'
        }

    def detect_language(self, text: str) -> str:
        url = f"{self.endpoint}/detect?api-version=3.0"
        body = [{"text": text}]
        response = requests.post(url, headers=self.headers, json=body)

        if response.status_code != 200:
            print("Error Response:", response.text)  # Print the full error message
            response.raise_for_status()

        detected_lang = response.json()[0]['language']
        return detected_lang

    def translate_to_english(self, text: str) -> str:
        """
        Translate the given text to English.
        :param text: Input text.
        :return: Translated text in English.
        """
        url = f"{self.endpoint}/translate?api-version=3.0&to=en"
        body = [{"text": text}]
        response = requests.post(url, headers=self.headers, json=body)
        response.raise_for_status()
        translated_text = response.json()[0]['translations'][0]['text']
        print("3C::", text, " <<<<translated to >>>>",translated_text)
        return translated_text

    def detect_and_translate(self, text: str) -> tuple:
        """
        Detect the language of the input text and translate it to English.
        :param text: Input text.
        :return: Tuple containing detected language and translated text.
        """
        detected_lang = self.detect_language(text)
        if detected_lang!="en":
            translated_text = self.translate_to_english(text)
        else:
            translated_text=text
            
        return detected_lang, translated_text