from model.engine import LLMAdapter

from typing import List
import re
import json
import logging
import time

class SemanticAdapter:
    def __init__(self, model):
        self.model:LLMAdapter=model
        self.MAX_RETRY=5
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
                    "role": "system",
                    "content": f"Generate a variation of the following user prompt "
                    f"by perturbing its semantics while preserving its core intent. "
                    f"Return just the string of the perturbed prompt.",
                }]
            )
        response, _ = self.model.complete(prompt)
        return response

    def sem_check(self, question_prompt, model_name):
        prompt=self.model.format_prompt(
        question_prompt,
        state=[{
                "role": "system",
                "content": f"You are a helpful assistant who is supposed to answer questions given to you. Keep the answer as " 
                f"concise as possible, does not have to be full sentences, just the answer is enough. " 
                f"For example, for the question: Who is the composer of Eine kleine Nachtmusik? Your response should be: Wolfgang Amadeus Mozart. ",   
            }]
        )
        response, _ = self.model.complete(prompt)
        return response
    
    def sem_perturb_combined(self, user_prompt, num_semantic) -> List[str]:
        prompt = (
            f"Generate variations of the following user prompt "
            f"by perturbing its semantics while preserving its core intent. Aim to create the most diverse "
            f"question set.  Respond with {num_semantic} perturbation(s) in json format. Please only respond with the json object. A sample is provided below for 2 perterbations: \n"
            f"<sample_question>When was Mozart born</sample_question>"
        )

        prompt += '<ans>{"perturbations": ["Care to share the birthdate of Mozart?", "Can you tell me the date when Mozart was born?"]}</ans>'
        prompt = self.model.format_prompt(
            user_prompt,
            state=[{
                    "role": "system",
                    "content":prompt
                }]
        )
        for retry_count in range(1, self.MAX_RETRY + 1):
            response, _ = self.model.complete(prompt)
            try:
                # Regex to find the JSON block
                json_match = re.search(r'{(.*)?}', response, re.DOTALL)
                if not json_match:
                    logging.info(f"No JSON object found in the provided text. retry: {retry_count}")
                json_data = json.loads(json_match.group())
                if "perturbations" not in json_data:
                    logging.info(f"The JSON does not contain the key 'perturbations'. retry: {retry_count}")
                perturbations = json_data["perturbations"]
                if len(perturbations) != num_semantic:
                    logging.info(f"length of perterbations is {len(perturbations)} instead of {num_semantic}. retry: {retry_count}")
                else:
                    return perturbations
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logging.info(f"Error encountered: {e}. retry: {retry_count}")
            time.sleep(0.3)
            
        raise RuntimeError(f"sem_perturb_combined: could not generate perturbations after {self.MAX_RETRY} retries")