from openai import OpenAI
from together import Together

import time
import json
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
together_client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))


class SemanticAdapter:
    def __init__(self):
        self.tree_creation_token_count = 0
        self.check_token_count = {}
        self.rag_token_count = 0
        pass

    def wiki_rag_completions(self, model_name, system_prompt, prompt):
        if model_name[:3] == "gpt":
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {"role": "user", "content": prompt},
                ],
            )
            self.rag_token_count += response.usage.total_tokens
            response_data = response.choices[0].message.content
            return response_data
        else:
            response = together_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": system_prompt + " " + prompt}
                ],
            )
            self.rag_token_count += response.usage.total_tokens
            return response.choices[0].message.content

    def sem_perturb(self, user_prompt):
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": f"Generate a variation of the following user prompt "
                    f"by perturbing its semantics while preserving its core intent. "
                    f"Return just the string of the perturbed prompt.",
                },
                {"role": "user", "content": user_prompt},
            ],
        )
        self.tree_creation_token_count += response.usage.total_tokens
        response_data = response.choices[0].message.content
        return response_data

    def sem_check(self, question_prompt, model_name="gpt-3.5-turbo"):
        if model_name[:3] == "gpt":
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant who is supposed to answer questions given to you. Keep the answer as concise as possible.",
                    },
                    {"role": "user", "content": question_prompt},
                ],
            )

            if model_name not in self.check_token_count:
                self.check_token_count[model_name] = response.usage.total_tokens
            self.check_token_count[model_name] += response.usage.total_tokens
            response_data = response.choices[0].message.content
            return response_data
        else:
            response = together_client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "user",
                        "content": "You are a helpful assistant who is supposed to answer questions given to you. Keep the answer as concise as possible. Question: "
                        + question_prompt,
                    }
                ],
                temperature=0.0,
            )
            self.check_token_count[model_name] = response.usage.total_tokens
            return response.choices[0].message.content

    def sem_perturb_combined(self, user_prompt, num_semantic):
        prompt = (
            f"Generate variations of the following user prompt "
            f"by perturbing its semantics while preserving its core intent. Aim to create the most diverse "
            f"question set.  Respond with {num_semantic} perturbation(s) in the json format as follows: "
        )

        prompt += '{perturbations: [""]}'
        while True:
            try:
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                self.tree_creation_token_count += response.usage.total_tokens
                response_data = json.loads(response.choices[0].message.content)
                perturbations = response_data["perturbations"]
                if len(perturbations) == num_semantic:
                    break
                else:
                    print(
                        "API returned a list of perturbations of the wrong length"
                    )
                    time.sleep(0.3)
            except Exception as e:
                print(user_prompt)
                print("API error: ", e)
        return perturbations
