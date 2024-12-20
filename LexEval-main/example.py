from tree.tree import Tree
from adapters.SemanticAdapter import SemanticAdapter
from adapters.SyntacticPerturb import SyntacticPerturber
from model.engine import GemmaAdapter
from adapters.rag import RAGAgent

import pandas as pd
import torch

import os
import logging
import time
from datetime import datetime
import gc
import sys
from collections import deque
import re

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

modelId ="google/gemma-2-9b-it"
RETRY_COUNT = 5
ERROR_THRESHOLD = 50
DLQ = []  # Dead Letter Queue
CHUNK_SIZE = 50
intermediatory_tree_dir="/vol/bitbucket/lst20/treenodes/2_2_1/tree/"

def process_with_retry(retry_count, index, row, model, semantic_adapter, syntactic_adapter, rag):
    try:
        logging.info(f"Processing row with index: {index}, attempt: {retry_count}")
        question = row["question"]
        possible_answers = row["possible_answers"]
        
        if not os.path.exists("trees/"):
            os.makedirs("trees/")
        tree_file_path = f"{intermediatory_tree_dir}/{index}.pkl"
        checked_tree_file_path = f"/vol/bitbucket/lst20/treenodes/2_2_1/complete/{index}_checked.pkl"
        
        # if exist file load
        test_tree = None
        if not os.path.exists(tree_file_path):
            test_tree = Tree(
                question,
                semantic_adapter,
                syntactic_adapter,
                rag
            )
            test_tree.make_tree(2, 2, 1, model_name=modelId)
            test_tree.save_tree(tree_file_path)
        
        # if tree not check  
        if not os.path.exists(checked_tree_file_path):    
            test_tree = Tree.load_tree(tree_file_path, semantic_adapter, syntactic_adapter, rag)
            test_tree.set_possible_answers(possible_answers)
            test_tree.run_check_pop_qa_batched(model_name=modelId, batch_size=5)
            test_tree.add_bleu_and_rouge(
                model_name=modelId
                )
            test_tree.save_tree(checked_tree_file_path)
            
        # remove intermediate tree
        if os.path.exists(tree_file_path):
            os.remove(tree_file_path)
        test_tree.print_tree()
        
    except ValueError as err:
        if "Make sure you have enough GPU RAM" in str(err):
            logging.error(f"OOM; attempt {retry_count}: {err}")
            del model
            gc.collect()
            torch.cuda.empty_cache()
            free_mem, total_mem = torch.cuda.mem_get_info()
            logging.info(f"after deleting model, {free_mem}/{total_mem} memory available")
            model = GemmaAdapter(modelId)
            semantic_adapter = SemanticAdapter(model)
            syntactic_adapter = SyntacticPerturber()
            rag = RAGAgent(model)
            process_with_retry(retry_count + 1, index, row, model, semantic_adapter, syntactic_adapter, rag)
    except RuntimeError as err:
        logging.error(f"RuntimeError encountered on attempt {retry_count}: {err}")
        if retry_count >= RETRY_COUNT:
            raise err
        #  delete model and re-instantiate
        del model
        gc.collect()
        torch.cuda.empty_cache()
        free_mem, total_mem = torch.cuda.mem_get_info()
        logging.info(f"after deleting model, {free_mem}/{total_mem} memory available")
        model = GemmaAdapter(modelId)
        semantic_adapter = SemanticAdapter(model)
        syntactic_adapter = SyntacticPerturber()
        process_with_retry(retry_count + 1, index, row, model, semantic_adapter, syntactic_adapter)

def get_tree_number(file_path):
    match = re.search(r'\d+', file_path)  # Finds the first occurrence of a number
    if match:
        first_number = match.group()
        return int(first_number)
    else:
        return -1

if __name__ == "__main__":
    # Check if the correct number of arguments is passed
    if len(sys.argv) < 3:
        print("wrong params!", sys.argv)
        sys.exit(1)
    formatted_time = datetime.fromtimestamp(time.time()).strftime('%m-%d-%H:%M:%S')
    filename = f"/vol/bitbucket/lst20/logs/{formatted_time}_logs.log"
    configure_logging(filename)
    
    file_deque = deque()
    # check where it last died
    for root, _, files in os.walk(intermediatory_tree_dir):
        for file in files:
            file_path = os.path.join(root, file)
            question_num = get_tree_number(file_path)
            if question_num >= 0:
                file_deque.append(question_num)
                
    start_idx = int(sys.argv[1])
    if len(file_deque) > 0:
        start_idx = min(int(sys.argv[1]), min(file_deque))
    end_idx = int(sys.argv[2])
    current_index = start_idx
    
    logging.info(f"processing dataset from {start_idx} to {end_idx}")

    gc.enable()
    
    #  read dataset
    # save test files
    df_location = "/vol/bitbucket/lst20/lex-eval_dataset/PopQA/test.csv"
    if not os.path.exists(df_location):
        # If the file does not exist, read the dataset from the source
        dir = os.path.dirname(df_location)
        if not os.path.exists(dir):
            os.makedirs(dir) 
        df = pd.read_csv("hf://datasets/akariasai/PopQA/test.tsv", sep='\t')
        df.to_csv(df_location, index=False)
        print(f"File downloaded and saved to {df_location}")
    
    start_time = time.time()
    
    logging.info("start time: ", start_time)
    model = GemmaAdapter(modelId)
    semantic_adapter = SemanticAdapter(model)
    syntactic_adapter = SyntacticPerturber()
    rag = RAGAgent(model)
    model_provisioned_time = time.time()
    model_start_duration = model_provisioned_time - start_time
    logging.info(f"client provisioned in: {model_start_duration:.3g} s")
    
    error_questions = 0
    for df_chunk in pd.read_csv(df_location, chunksize=CHUNK_SIZE, usecols=['question', 'possible_answers']):
        # print(end_idx, df_chunk.index[-1], current_index, df_chunk.index[0])
        if current_index >= end_idx:
            break
        
        if end_idx < df_chunk.index[-1] or current_index > df_chunk.index[0]:
            lower_bound = max(start_idx, df_chunk.index[0])
            upper_bound = min(end_idx, df_chunk.index[-1])
            df_chunk = df_chunk[lower_bound:upper_bound + 1]
            current_index = upper_bound
        else:
            current_index = df_chunk.index[-1]
            
        for index, row in df_chunk.iterrows():
            try:
                process_with_retry(0, index, row, model, semantic_adapter, syntactic_adapter, rag)
            except Exception or RuntimeError as err:
                if error_questions < ERROR_THRESHOLD:
                    error_questions += 1
                    logging.info(f"Qeustion of index: {index} cannot be processed, adding to DLQ. {ERROR_THRESHOLD - error_questions} tries left. error: {err}")
                    DLQ.append({"num": index, "err": err})
                else:
                    logging.error(f"Error threshold reached. Question after index {index} will not be processed.")
                    DLQ.append({"num": index, "err": err})
                    print("DLQ: ", DLQ)
                    raise err
            
    end_time = time.time()
    time_taken = time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))
    logging.info(f"done! time to evaluate {end_idx - start_idx + 1} trees: {time_taken}")
    print(f"done! time to evaluate {end_idx - start_idx + 1} trees: {time_taken}")
    print("DLQ: ", DLQ)