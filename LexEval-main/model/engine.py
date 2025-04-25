import re
import gc
from typing import Dict, List, Optional
import time
from datetime import datetime
import random
import threading

import torch
from transformers import LlamaForCausalLM, Gemma3ForCausalLM, GenerationConfig, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import boto3
import botocore 
import json
import logging


gc.enable()

class LLMAdapter:
    """Generic adapter class for all LLMs
    """
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def __del__(self):
        try:
            self.cleanup()
        except Exception as e:
            print(f"Error during __del__: {e}")
            
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
        
    def cleanup(self):
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

        self.pipeline = pipeline("text-generation",
                                 model=self.model,
                                 tokenizer=self.tokenizer,
                                 pad_token_id=self.tokenizer.eos_token_id,
                                 max_new_tokens=256
                                )
        
    def cleanup(self):
        self.quantization_options = None
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        gc.collect()
        torch.cuda.empty_cache()
         
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
            if temperature is not None and temperature > 0:
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


class Gemma3Adapter(LLMAdapter):
    '''
     For the 1B text only model and to load thevision language models
     like they were language models (omitting the vision tower)
    '''
    def __init__(self, model_path, **kwargs) -> None:
        # Define the quantization configuration
        self.quantization_options = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        # Load the model with quantization
        self.model = Gemma3ForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            quantization_config=self.quantization_options
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Ensure left‑padding so that the attention mask aligns with Flash‑Attention
        # blocks; we also guarantee a pad token exists.
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def cleanup(self):
            self.quantization_options = None
            self.model = None
            self.tokenizer = None
            gc.collect()
            torch.cuda.empty_cache()
            
    def format_prompt(self, utterance: str, state: List[Dict[str, str]] = None, role='user', **kwargs):
        prompts = []

        formatted_conversation = []

        # Process the state messages (e.g. system prompt)
        if state is not None:
            for message in state:
                formatted_conversation.append({
                    "role": message["role"],
                    "content": [{"type": "text", "text": message["content"]}]
                })

        # Add the user prompt as a user message
        formatted_conversation.append({
            "role": role,
            "content": [{"type": "text", "text": utterance}]
        })

        prompts.append(formatted_conversation)
        return prompts
    

    def complete(self, prompt: List[Dict[str, str]], **kwargs):
        temperature = kwargs.get('temperature', None)  # Default beam search instead of sampling
        with torch.no_grad(): 
            if temperature is not None and temperature > 0:
                gen_config = GenerationConfig(
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=temperature,
                    num_beams=2,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            else:
                gen_config = GenerationConfig(
                    max_new_tokens=256,
                    num_beams=2,
                    do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id,
                )   
            inputs = self.tokenizer.apply_chat_template(
                prompt, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt"
            ).to(self.model.device)
            input_len = inputs["input_ids"].shape[-1]
            with torch.inference_mode():
                generation = self.model.generate(**inputs, generation_config=gen_config)
                generation = generation[0][input_len:]
            response = self.tokenizer.decode(generation, skip_special_tokens=True)
            if type(response) is list:
                response = response[-1]['content']
            new_state = prompt[1:] + self.format_prompt(response, role='assistant')
            gc.collect()
            torch.cuda.empty_cache()

        return response, new_state
  
    
class Llama32Adapter(LLMAdapter):
    def __init__(self, model_path, **kwargs) -> None:
        self.quantization_options = BitsAndBytesConfig(load_in_4bit=True,
                                                       bnb_4bit_quant_type='nf4',
                                                       bnb_4bit_use_double_quant=True,
                                                       bnb_4bit_compute_dtype=torch.bfloat16
                                                       )
        self.model = LlamaForCausalLM.from_pretrained(model_path,
                                                      quantization_config=self.quantization_options,
                                                        device_map='auto'
        )
        
        # setup tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
        self.pipeline = pipeline("text-generation",
                                 model=self.model,
                                 tokenizer=self.tokenizer,
                                 pad_token_id=self.tokenizer.eos_token_id,
                                 max_new_tokens=512
                                )

    def cleanup(self):
        self.quantization_options = None
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        gc.collect()
        torch.cuda.empty_cache()
        
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
     
     
class MistralInstructAdapter(LLMAdapter):
    def __init__(self, model_path, **kwargs) -> None:
        # Configure quantization for 4-bit inference
        self.quantization_options = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=self.quantization_options,
            device_map='auto'
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
    def cleanup(self):
        self.quantization_options = None
        self.model = None
        self.tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()   
             
    def format_prompt(self, utterance: str, state: List[Dict[str, str]] = None, role='user',) -> str:
        prompts = []

        formatted_conversation = []

        # Process the state messages (a.k.a. system prompt)
        if state is not None:
            for message in state:
                formatted_conversation.append({
                    "role": message["role"],
                    "content":
                        message["content"] + "\n" +
                        "IMPPORTANT: Do not explain your reasoning / add additional notes, if not asked for. THIS IS VERY IMPORTANT!!"
         
                })

        # Add the user prompt as a user message
        formatted_conversation.append({
            "role": role,
            "content": utterance
        })

        prompts.append(formatted_conversation)
        return prompts
    

    def complete(self, formatted_prompt: List[Dict[str, str]], guidelines: List[Dict[str, str]] = None, **kwargs):
        temperature = kwargs.get('temperature', None)

        if temperature is not None and temperature > 0:
            gen_config = GenerationConfig(
                max_new_tokens=256,
                do_sample=True,
                temperature=temperature,
                num_beams=5,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        else:
            gen_config = GenerationConfig(
                max_new_tokens=256,
                pad_token_id=self.tokenizer.eos_token_id,
            )  

        formatted_prompt = self.tokenizer.apply_chat_template(
            formatted_prompt, return_tensors="pt", tokenize=False
        )
        model_input = self.tokenizer(formatted_prompt, return_tensors="pt", padding=True).to(self.model.device)
        attention_mask = model_input["attention_mask"]
        input_len = model_input['input_ids'].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(model_input['input_ids'],
                                             attention_mask=attention_mask,
                                             generation_config=gen_config)
            generation = generation[0][input_len:-1]
        decoded_output = self.tokenizer.batch_decode([generation])[0]
        new_state = formatted_prompt + [{"role": "assistant", "content": decoded_output}]
        gc.collect()
        torch.cuda.empty_cache()

        return decoded_output, new_state
        
        
class MistralInstructAwsAdapter(LLMAdapter):
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

    def cleanup(self):
        pass
        
    def format_prompt(self, utterance: str, state: List[Dict[str, str]] = None, role='user', **kwargs):
        """
        Format the prompt following an instruct style.
        If guidelines are provided, they are included as context.
        """
        prompt = ""
        if state:
            prompt += "\nInstruction:\n"
            prompt += "\n".join(f"- {s['content']}" for s in state)
            prompt += "\n"
        prompt += "IMPPORTANT: Do not explain your reasoning for tasks that does not ask for it!!!!"
        prompt += f"\n### Question:\n{utterance}\n\n### Response:\n"
        return prompt

    def complete(self, prompt: List[Dict[str, str]], **kwargs):
        with self.lock:  # Synchronize access
            self.requests += 1
            if self.requests % 10 == 0:
                current_time = time.time()
                formatted_time = datetime.fromtimestamp(current_time).strftime('%m-%d-%H:%M:%S')
                logging.info(f"{formatted_time}: {self.requests} request sent to bedrock client")
            
        request_param = {
            "prompt": prompt,
            "temperature": 0,
            "max_tokens": 200
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
                        body=body, modelId=self.model_path, accept=accept, contentType=contentType
                    )
                    response_body = json.loads(response.get("body").read())
                    ans = response_body['outputs'][0]['text']
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