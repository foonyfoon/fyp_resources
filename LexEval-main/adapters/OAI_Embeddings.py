import os
import requests


class OAIEmbedAdapter:
    def __init__(self):
        self.EMBED_URL = "https://api.openai.com/v1/embeddings"
        self.MODEL_ADA = "text-embedding-ada-002"
        self.api_key = os.getenv("NOMAD_AI_API_KEY")

    # @property
    # def __dict__(self):
    #     return {
    #         "MODEL_ADA": self.MODEL_ADA,
    #     }

    def encode(self, prompt):
        if type(prompt) == dict:
            prompt = str(prompt)

        data = {
            "input": prompt,
            "model": 'text-embedding-ada-002'
        }
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.api_key,
        }

        response = requests.post(self.EMBED_URL, json=data, headers=headers)

        if response.status_code == 200:
            response_data = response.json()['data'][0]['embedding']
            return response_data
            # print(response_data)
        else:
            print(response.json())
            print("embedding request failed")
