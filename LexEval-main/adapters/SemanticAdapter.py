from model.engine import GemmaAdapter

from typing import List
import re
import json

class SemanticAdapter:
    def __init__(self, model):
        self.model:GemmaAdapter=model
        pass

    def wiki_rag_completions(self, model_name, system_prompt, user_prompt):
        prompt=self.model.format_prompt(
            user_prompt,
            state=[{
                    "role": "system",
                    "content": system_prompt,
                }]
            )
        response, _ = self.model.complete(prompt)
        return response


    def sem_perturb(self, user_prompt):
        prompt=self.model.format_prompt(
            user_prompt,
            state=[{
                    "role": "user",
                    "content": f"Generate a variation of the following user prompt "
                    f"by perturbing its semantics while preserving its core intent. "
                    f"Return just the string of the perturbed prompt.",
                }]
            )
        response, _ = self.model.complete(prompt)
        return response

    def sem_check(self, question_prompt, model_name="gemma"):
        prompt=self.model.format_prompt(
        question_prompt,
        state=[{
                "role": "system",
                "content": "You are a helpful assistant who is supposed to answer questions given to you. Keep the answer as concise as possible.",
            }]
        )
        response, _ = self.model.complete(prompt)
        return response
    
    def sem_perturb_combined(self, user_prompt, num_semantic) -> List[str]:
        prompt = (
            f"Generate variations of the following user prompt "
            f"by perturbing its semantics while preserving its core intent. Aim to create the most diverse "
            f"question set.  Respond with {num_semantic} perturbation(s) in json format. For example, for the question"
            f"When was Mozart born, the response for 2 perterbation is as follows: "
        )

        prompt += '{"perturbations": ["Care to share the birthdate of Mozart?", "Can you tell me the date when Mozart was born?"]}'
        prompt = self.model.format_prompt(
            user_prompt,
            state=[{
                    "role": "system",
                    "content":prompt
                }]
        )
        response, _ = self.model.complete(prompt)
        try:
            # Regex to find the JSON block
            json_match = re.search(r'{(.*)?}', response, re.DOTALL)
            if not json_match:
                raise ValueError("No JSON object found in the provided text.")
            json_data = json.loads(json_match.group())
            if "perturbations" not in json_data:
                raise KeyError("The JSON does not contain the key 'perturbations'.")
            perturbations = json_data["perturbations"]
            return perturbations
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            raise e