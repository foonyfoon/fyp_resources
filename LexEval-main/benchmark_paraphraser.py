# helper functions

import pickle
import os
import sys
import pandas as pd
import time
import logging
from datetime import datetime

from adapters.SemanticPerturb import ParaphrasePerturber
from model.engine import LLMAdapter
from similarity.cosine_similarity import similarity
from textstat import flesch_kincaid_grade, dale_chall_readability_score
from textdistance import levenshtein
from adapters.OAI_Embeddings import EmbedAdapter, RobertaEmbedder

def configure_logging(filename):
    dir = os.path.dirname(filename)
    if not os.path.exists(dir):
        os.makedirs(dir)
    # Configure logging
    logging.basicConfig(
        filename=filename,
        filemode='a',
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO
    )
    
# code to load semantic adapter
def get_paraphraser(paraphrase_modelId: str) -> ParaphrasePerturber:
    model: LLMAdapter = None
    if "gemma-3" in paraphrase_modelId.lower():
        from model.engine import Gemma3Adapter
        model = Gemma3Adapter(paraphrase_modelId)
    elif "gemma" in paraphrase_modelId.lower():
        from model.engine import GemmaAdapter
        model = GemmaAdapter(paraphrase_modelId)
    elif "mistralai" in paraphrase_modelId.lower():
        from model.engine import MistralInstructAdapter
        model = MistralInstructAdapter(paraphrase_modelId)
    elif "mistral.mistral-7b-instruct-v0:2" in paraphrase_modelId.lower():
        from model.engine import MistralInstructAwsAdapter
        model = MistralInstructAwsAdapter(paraphrase_modelId)
    else:
        raise NotImplementedError(f"No adapter implemented for: {paraphrase_modelId}")

    return ParaphrasePerturber(model)

def preturb(paraphraser: ParaphrasePerturber, root_prompt: str, max_perturb: int, encoder: EmbedAdapter) -> list:
    prompt_list = [root_prompt]
    perturbations = []
    max_retries = 5
    max_temp = 1.5
    root_embedding = encoder.encode(root_prompt)

    for i in range(max_perturb):
        retry_count = 0
        is_valid = False
        current_prompt = root_prompt
        selected_perturbation = None

        start_time = time.time()
        while not is_valid and retry_count < max_retries:
            temp = min(max_temp, 1.5 * ((1 + retry_count) / max_retries))

            perturbation = paraphraser.sem_perturb(current_prompt, prompt_list, temp)
            perturbation = perturbation.strip()

            is_valid = (
                perturbation not in prompt_list
            )

            if is_valid:
                selected_perturbation = perturbation
                prompt_list.append(perturbation)
            else:
                retry_count += 1
                if perturbation in prompt_list:
                    prompt_list.remove(perturbation)
                    prompt_list.insert(0, perturbation)

        if not is_valid:
            logging.info(f"preturb: Failed to generate valid perturbation after {retry_count} retries.")
            continue
        
        end_time = time.time()

        # Compute metrics
        fk_score = flesch_kincaid_grade(selected_perturbation)
        dc_score = dale_chall_readability_score(selected_perturbation)
        complexity_score = (fk_score + dc_score) / 2
        lev_distance = levenshtein.distance(root_prompt, selected_perturbation)
        similarity_score = similarity(root_embedding, encoder.encode(selected_perturbation))    

        perturbations.append({
            "is_valid": is_valid,
            "text": selected_perturbation,
            "similarity_score": similarity_score,
            "levenshtein_distance": lev_distance,
            "flesch_kincaid": fk_score,
            "dale_chall": dc_score,
            "complexity_score": complexity_score,
            "seconds": end_time - start_time,
        })
    return perturbations


def save_progress(progress_status, perturbation_result, progress_file, result_file):
    with open(progress_file, "wb") as f:
        pickle.dump(progress_status, f)
    with open(result_file, "wb") as f:
        pickle.dump(perturbation_result, f)
        
if __name__ == "__main__":
    # preturb until it repeats or exceeds max perturbations
    if len(sys.argv) != 2:
        end_idx = 50
        logging.info("end_idx = 50")
    else:
        end_idx = int(sys.argv[1])

    formatted_time = datetime.fromtimestamp(time.time()).strftime('%m-%d-%H:%M-%S')
    filename = f"/vol/bitbucket/lst20/preturb_benchmark/benchmark_paraphraser_{formatted_time}_logs.log"
    configure_logging(filename)
    MAX_PERTURB = 20

    # Set random seed
    SEED = 42
    NUM_SAMPLES = 100

    # Load the dataset
    DF_LOCATION = "/vol/bitbucket/lst20/lex-eval_dataset/PopQA/test.csv"
    SAMPLE_FILE = "/vol/bitbucket/lst20/preturb_benchmark/sample.csv"
    PROGRESS_FILE = "/vol/bitbucket/lst20/preturb_benchmark/progress_status.pkl"
    RESULT_FILE = "/vol/bitbucket/lst20/preturb_benchmark/perturbation_result.pkl"

    if not os.path.exists(SAMPLE_FILE):
        df = pd.read_csv(DF_LOCATION, usecols=['question'])
        sampled_df = df.sample(n=NUM_SAMPLES, random_state=SEED).reset_index(drop=True)
        os.makedirs(os.path.dirname(SAMPLE_FILE), exist_ok=True)
        sampled_df.to_csv(SAMPLE_FILE, index=False)
    else:
        sampled_df = pd.read_csv(SAMPLE_FILE)
        
    embedder = RobertaEmbedder()
    model_contenders = [
        "mistralai/Mistral-7B-Instruct-v0.2",
        "google/gemma-3-4b-it",
        "google/gemma-3-12b-it",
    ]

    # For each model, progress_status stores indices already processed.
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'rb') as f:
            progress_status = pickle.load(f)
        logging.info("Loaded progress_status from pickle file.")
    else:
        progress_status = {model_id: {} for model_id in model_contenders}
        logging.info("Initialized empty progress_status.")

    # For each model, perturbation_result stores a dictionary mapping row index to the result (list).
    if os.path.exists(RESULT_FILE):
        with open(RESULT_FILE, 'rb') as f:
            perturbation_result = pickle.load(f)
        logging.info("Loaded perturbation_result from pickle file.")
    else:
        perturbation_result = {model_id: {} for model_id in model_contenders}
        logging.info("Initialized empty perturbation_result.")
        
    for model_id in model_contenders:
        with get_paraphraser(model_id) as perturber:
            # if model not in progress_status, initialize it
            if model_id not in progress_status:
                progress_status[model_id] = {}
                perturbation_result[model_id] = {}
                logging.info(f"Initialized progress status and perturbation result for model {model_id}.")
            # Iterate through the sampled rows
            for index, row in sampled_df.iloc[:end_idx].iterrows():
                if perturbation_result[model_id].get(index) is not None:
                    continue  # Skip if already processed
                question = row['question']
                perturbations = preturb(perturber, question, MAX_PERTURB, embedder)
                # Update tracking dictionaries.
                progress_status[model_id][index] = True  # Mark this index as processed.
                perturbation_result[model_id][index] = perturbations
                # save progress status
                save_progress(progress_status, perturbation_result, PROGRESS_FILE, RESULT_FILE)
                logging.info(f"Processed index {index} for model {model_id}.")
            
                