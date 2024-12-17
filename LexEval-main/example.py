from tree.tree import Tree
from adapters.SemanticAdapter import SemanticAdapter
from adapters.SyntacticPerturb import SyntacticPerturber
from model.engine import GemmaAdapter

import pandas as pd
import torch

import os
import logging
import time
import gc
import sys

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

RETRY_COUNT = 5
ERROR_THRESHOLD = 8
DLQ = []  # Dead Letter Queue

def process_with_retry(retry_count, index, row, model, semantic_adapter, syntactic_adapter):
    try:
        logging.info(f"Processing row with index: {index}, attempt: {retry_count}")
        question = row["question"]
        possible_answers = row["possible_answers"]
        
        if not os.path.exists("trees/"):
            os.makedirs("trees/")
        tree_file_path = f"/vol/bitbucket/lst20/treenodes/2_2_1/{index}.pkl"
        checked_tree_file_path = f"/vol/bitbucket/lst20/treenodes/2_2_1/{index}_checked.pkl"
        # if exist file load
        test_tree = None
        if not os.path.exists(tree_file_path):
            test_tree = Tree(
                question,
                semantic_adapter,
                syntactic_adapter
            )
            test_tree.make_tree(2, 2, 1, model_name='gpt-3.5-turbo')
            test_tree.save_tree(tree_file_path)
        test_tree = Tree.load_tree(tree_file_path, semantic_adapter, syntactic_adapter)
        test_tree.set_possible_answers(possible_answers)
        test_tree.run_check_pop_qa_batched(model_name="google/gemma-7b-it", batch_size=5)
        test_tree.add_bleu_and_rouge(
            model_name='google/gemma-7b-it'
            )
        test_tree.save_tree(checked_tree_file_path)
        test_tree.print_tree()
    except RuntimeError as err:
        logging.error(f"RuntimeError encountered on attempt {retry_count}: {err}")
        if retry_count >= RETRY_COUNT:
            raise err
        #  delete model and re-instantiate
        del model
        gc.collect()
        torch.cuda.empty_cache()
        free_mem, total_mem = torch.cuda.mem_get_info()
        print(f"after deleting model, {free_mem}/{total_mem} memory available")
        model = GemmaAdapter("google/gemma-2-9b-it")
        semantic_adapter = SemanticAdapter(model)
        syntactic_adapter = SyntacticPerturber()
        process_with_retry(retry_count + 1, index, row, model, semantic_adapter, syntactic_adapter)

if __name__ == "__main__":
    # Check if the correct number of arguments is passed
    if len(sys.argv) < 3:
        print("wrong params!", sys.argv)
        sys.exit(1)

    start_idx = int(sys.argv[1])
    end_idx = int(sys.argv[2])

    gc.enable()
    
    #  read dataset
    df = pd.read_csv("hf://datasets/akariasai/PopQA/test.tsv", sep="\t", usecols=["question", "possible_answers"])
    free_mem, total_mem = torch.cuda.mem_get_info()
    print(f"{free_mem}/{total_mem} memory available")      
    if start_idx > end_idx or end_idx > df.shape[0]:
        raise ValueError("wrong value for start_idx or end_idx")
    
    df = df[start_idx:end_idx]
    start_time = time.time()
    logging.info("start time: ", start_time)
    model = GemmaAdapter("google/gemma-2-9b-it")
    semantic_adapter = SemanticAdapter(model)
    syntactic_adapter = SyntacticPerturber()
    error_questions = 0

    for index, row in df.iterrows():
        free_mem, total_mem = torch.cuda.mem_get_info()
        logging.info(f"{free_mem}/{total_mem} memory available")
        try:
            process_with_retry(0, index, row, model, semantic_adapter, syntactic_adapter)
        except RuntimeError as err:
            if error_questions < ERROR_THRESHOLD:
                error_questions += 1
                logging.info(f"Qeustion of index: {index} cannot be processed, adding to DLQ. {ERROR_THRESHOLD - error_questions} tries left")
            else:
                logging.error(f"Error threshold reached. Question after index {index} will not be processed.")
                DLQ.append((index, row, str(err)))
                print("DLQ: ", DLQ)
                raise err
                
    end_time = time.time()
    logging.info(f"done! time to evaluate {end_idx - start_idx} trees: ", end_time - start_time)     
