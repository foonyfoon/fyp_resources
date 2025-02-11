import re
import gc
from typing import Dict, List, Optional
import time
from datetime import datetime
import random
import threading

import torch
from transformers import LlamaForCausalLM, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import boto3
import botocore 
import json
import logging


gc.enable()

class LLMAdapter:
    """Generic adapter class for all LLMs
    """
    def format_prompt(self, utterance: str, state: List[Dict[str, str]]=None, role='user', **kwargs):
        """Given a conversation state and user utterance, format the utterance into a prompt format for the given LLM

        Args:
            state (List[Dict[str, str]]): _description_
            utterance (str): _description_

        Returns:
            List[Dict[str, str]]: completed prompt
        """
        raise NotImplementedError()

    def complete(self, prompt: List[Dict[str, str]], **kwargs):
        """Given a prompt, provide a completion response

        Args:
            prompt (List[Dict[str, str]]): prompt given to the LLM model to generate a response

        Returns:
            Tuple[str, List[Dict[str, str]]]: response string and new state with response
        """
        raise NotImplementedError()


class GemmaAdapter(LLMAdapter):
    def __init__(self, model_path, **kwargs) -> None:
        self.quantization_options = BitsAndBytesConfig(load_in_4bit=True,
                                                       bnb_4bit_quant_type='nf4',
                                                       bnb_4bit_use_double_quant=True,
                                                       bnb_4bit_compute_dtype=torch.bfloat16
                                                       )
        self.model = AutoModelForCausalLM.from_pretrained(model_path,
                                               device_map='auto',
                                               quantization_config=self.quantization_options,
                                               **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.generation_config = {
            "repetition_penalty": 1.1,
            "pad_token_id": self.tokenizer.eos_token_id
        }
        self.pipeline = pipeline("text-generation",
                                 model=self.model,
                                 tokenizer=self.tokenizer,
                                 pad_token_id=self.tokenizer.eos_token_id,
                                 max_new_tokens=256
                                )

    def format_prompt(self, utterance: str, state: List[Dict[str, str]] = None, role='user', **kwargs):
        prompt = []
        instr = ""
        if state is not None:
            instr += "Here are the guidelines for answering questions: \n"
            instr += "\n".join(f"{s['content']}" for i, s in enumerate(state))
            instr += "\nNow, here is the question you need to answer: \n"
        instr += utterance
        prompt.append({
            'role': role,
            'content': instr
        })
        return prompt

    def complete(self, prompt: List[Dict[str, str]], **kwargs):
        temperature = kwargs.get('temperature', None)  # Default beam search instead of sampling
        with torch.no_grad(): 
            if temperature is not None:
                obj_response = self.pipeline(prompt, temperature=temperature, do_sample=True)
            else:
                obj_response = self.pipeline(prompt)
            response = obj_response[0]['generated_text']
            if type(response) is list:
                response = response[-1]['content']
            new_state = prompt[1:] + self.format_prompt(response, role='assistant')
            gc.collect()
            torch.cuda.empty_cache()

        return response, new_state


class ClaudeAdapter(LLMAdapter):
    def __init__(self, model_path, **kwargs) -> None:
        self.input_tokens = 0
        self.output_tokens = 0
        self.requests = 0
        self.model_path = model_path
        boto_session = boto3.session.Session()
        self.MAX_RETRIES = 10
        self.BASE_DELAY = 15
        self.MAX_DELAY = 180
        self.bedrock_runtime_client = boto_session.client(
                service_name='bedrock-runtime',
                region_name=boto_session.region_name,
            )
        self.lock = threading.Lock()

    def format_prompt(self, utterance: str, state: List[Dict[str, str]] = None, role='user', **kwargs):
        instr = ""
        if state is not None:
            instr += "<instruction> \n"
            instr += "\n".join(f"{i + 1}. {s['content']}" for i, s in enumerate(state))
            instr += "\n</instruction>\n"
        instr += f"\n<question>\n{utterance}\n</question>"
        return instr

    def complete(self, prompt: List[Dict[str, str]], **kwargs):
        with self.lock:  # Synchronize access
            self.requests += 1
            if self.requests % 10 == 0:
                current_time = time.time()
                formatted_time = datetime.fromtimestamp(current_time).strftime('%m-%d-%H:%M:%S')
                logging.info(f"{formatted_time}: {self.requests} request sent to bedrock client")
            
        request_param = {
            "temperature": 0,
            "max_tokens": 200,
            "anthropic_version": "bedrock-2023-05-31",
            "messages": [{"role": "user", "content": prompt}],
        }   
        body = json.dumps(request_param)
        accept = "application/json"
        contentType = "application/json"

        attempt = 0
        response = None
        while True:
            if attempt >= self.MAX_RETRIES:
                raise RuntimeError("ThrottlingException: You have sent too many requests, exceeded retry limit")
            else:
                try:
                    response = self.bedrock_runtime_client.invoke_model(
                        body=body, modelId='anthropic.claude-3-haiku-20240307-v1:0', accept=accept, contentType=contentType
                    )
                    response_body = json.loads(response.get("body").read())
                    input_tokens = response_body['usage']['input_tokens']
                    output_tokens = response_body['usage']['output_tokens']
                    with self.lock:
                        self.input_tokens += input_tokens
                        self.output_tokens += output_tokens
                    ans = response_body['content'][0]['text']
                    new_state = prompt[1:] + self.format_prompt(ans, role='assistant')
                    return ans, new_state
                
                except self.bedrock_runtime_client.exceptions.ThrottlingException as e:
                    attempt += 1
                    # Calculate delay with exponential backoff and jitter
                    delay = min(self.BASE_DELAY * 2 ** attempt, self.MAX_DELAY)
                    delay += random.uniform(0, 1)
                    logging.info(f"usage: requests={self.requests}, input_tokens={self.input_tokens}, output_tokens={self.output_tokens}")
                    logging.info(f"Request throttled. Retrying in {delay:.2f} seconds... (Attempt {attempt}/{self.MAX_RETRIES})")
                    time.sleep(delay)
                except botocore.exceptions.ClientError as e:
                    # Handle other errors
                    attempt += 1
                    print(f"Bedrock ClientError: {e}")
                    raise e
                except botocore.exceptions.ClientError as error:
                    attempt += 1
                    if error.response['Error']['Code'] == 'AccessDeniedException':
                        raise RuntimeError("AccessDeniedException: please check if you have access to bedrock or the model specified.")
                    else:
                        raise error