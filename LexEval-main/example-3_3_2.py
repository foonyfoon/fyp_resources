from tree.tree import Tree
from adapters.SemanticAdapter import SemanticAdapter
from adapters.SemanticPerturb import ParaphrasePerturber
from adapters.SyntacticPerturb import SyntacticPerturber
from adapters.OAI_Embeddings import EmbedAdapter, RobertaEmbedder
from model.engine import GemmaAdapter

import pandas as pd
import torch

import os
import logging
import time
from datetime import datetime
import gc
import sys
import re
import traceback

RETRY_COUNT = 2
ERROR_THRESHOLD = 3000
DLQ = []  # Dead Letter Queue
CHUNK_SIZE = 50
intermediatory_tree_dir="/vol/bitbucket/lst20/treenodes/base/3_1_0/tree"
checked_tree_dir="/vol/bitbucket/lst20/treenodes/base/3_1_0/complete"

def clear_cache():
    # Clear any leftover tensors
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj):
                if obj.is_cuda:
                    del obj
        except Exception:
            pass
    gc.collect()
    torch.cuda.empty_cache()
            
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
    
def write_to_dlq(filename, text):
    with open(filename, 'a') as dlq_file:
        dlq_file.write(f"{text}:\n")   


def get_answer(index, generator, modelId, embedder):
    tree_file_path = f"{intermediatory_tree_dir}/{index}.pkl"
    checked_tree_file_path = f"{checked_tree_dir}/{index}_checked.pkl"
    max_retries = RETRY_COUNT
    for retry_count in range(max_retries + 1):
        logging.info(f"Evaluating tree of dataset row: {index}, attempt: {retry_count}")
        try:
            # Load tree in evaluation mode using the generator
            test_tree = Tree.load_tree(tree_file_path, eval=True, generator=generator, embedder=embedder)
            test_tree.run_check_pop_qa_batched(model_name=modelId)
            test_tree.add_bleu_and_rouge(model_name=modelId)
            test_tree.save_tree(checked_tree_file_path)
            test_tree.print_tree()
            
            # Explicitly remove the tree if it's no longer needed
            del test_tree
            break  # Success: exit the retry loop
        
        except ValueError as err:
            if "Make sure you have enough GPU RAM" in str(err):
                logging.error(f"OOM; attempt {retry_count}: {err}")
                clear_cache()
                free_mem, total_mem = torch.cuda.mem_get_info()
                logging.info(f"After clearing cache: {free_mem}/{total_mem} memory available")
            else:
                raise err  # re-raise if it’s a different ValueError
        
        except RuntimeError as err:
            logging.error(f"RuntimeError encountered on attempt {retry_count}: {err}")
            if retry_count == max_retries:
                raise err  # re-raise if max retries reached


def get_question_tree(index, row, semantic_adapter, syntactic_adapter, ner_model, embedder):
    max_retries = RETRY_COUNT
    for retry_count in range(max_retries + 1):
        try:
            logging.info(f"creating tree of dataset row: {index}, attempt: {retry_count}")
            question = row["question"]
            possible_answers = row["possible_answers"]

            if not os.path.exists("trees/"):
                os.makedirs("trees/")
            tree_file_path = f"{intermediatory_tree_dir}/{index}.pkl"
            checked_tree_file_path = f"{checked_tree_dir}/{index}_checked.pkl"

            # Only create a tree if not already saved
            if not os.path.exists(tree_file_path) and not os.path.exists(checked_tree_file_path):
                test_tree = Tree(
                    question,
                    eval=False,
                    sem_perturber=semantic_adapter,
                    syn_perturber=syntactic_adapter,
                    ner_model=ner_model,
                    embedder=embedder
                )
                test_tree.make_tree(3, 1, 0)
                test_tree.set_possible_answers(possible_answers)
                test_tree.save_tree(tree_file_path)
                
                # Optionally, delete test_tree if it won't be used again:
                del test_tree
            # Exit the loop if successful
            break

        except ValueError as err:
            if "Make sure you have enough GPU RAM" in str(err):
                logging.error(f"OOM; attempt {retry_count}: {err}")
                clear_cache()
                free_mem, total_mem = torch.cuda.mem_get_info()
                logging.info(f"After clearing cache: {free_mem}/{total_mem} memory available")
            else:
                raise err
        except RuntimeError as err:
            logging.error(f"RuntimeError encountered on attempt {retry_count}: {err}")
            if retry_count == max_retries:
                raise err


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
    
    # create dlq file
    dlq_path = f"/vol/bitbucket/lst20/dlq/{formatted_time}_dlq.log"
    dlq_dir = os.path.dirname(dlq_path)
    if not os.path.exists(dlq_dir):
        os.makedirs(dlq_dir)
    with open(dlq_path, 'a') as dlq_file:
        dlq_file.write("DLQ:\n")
    

    start_idx = int(sys.argv[1])
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
    logging.info("Start time: %s", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)))
    ##################### factory this ###########################
    preturb_modelId = "google/gemma-2-9b-it"
    model = GemmaAdapter(preturb_modelId)
    semantic_adapter = ParaphrasePerturber(model)
    syntactic_adapter = SyntacticPerturber()
    embedder = RobertaEmbedder()
    ##############################################################
    model_provisioned_time = time.time()
    model_start_duration = model_provisioned_time - start_time
    logging.info(f"model ready in: {model_start_duration:.3g} s")
    
    error_questions = 0
    df = pd.read_csv(df_location, usecols=['question', 'possible_answers'])
    # generate questions
    for index, row in df.iloc[start_idx:end_idx + 1].iterrows():
        try:
            with torch.no_grad():
                get_question_tree(index, row, semantic_adapter, syntactic_adapter, model, embedder)
        except Exception or RuntimeError as err:
            if error_questions < ERROR_THRESHOLD:
                error_questions += 1
                # Get the full stack trace as a string
                stack_trace = traceback.format_exc()
                logging.info(
                    f"Question of index: {index} cannot be processed, adding to DLQ. "
                    f"{ERROR_THRESHOLD - error_questions} tries left. error: {err}\nStack trace:\n{stack_trace}"
                )
                error_entry = {"num": index, "err": str(err), "trace": stack_trace}
                DLQ.append(error_entry)
                write_to_dlq(dlq_path, str(error_entry))
            else:
                logging.error(f"Error threshold reached. Question after index {index} will not be processed.")
                DLQ.append({"num": index, "err": err})
                print("DLQ: ", DLQ)
                write_to_dlq(dlq_path, str({"num": index, "err": err}))
                raise err
            
    # clear cache
    semantic_adapter.model = None
    del semantic_adapter
    del syntactic_adapter
    del model
    clear_cache()
    torch.cuda.synchronize()
    
    free_mem, total_mem = torch.cuda.mem_get_info()
    logging.info(f"after clearing cache, {free_mem}/{total_mem} memory available")
    logging.info(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    logging.info(f"Cached:    {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

    
    logging.info("********* start eval tree *********")
    ##################### factory this ###########################
    gen_modelId = "google/gemma-2-9b-it"
    generator_model = GemmaAdapter(gen_modelId)
    generator = generator_model
    embedder = RobertaEmbedder()
    # ############################################################
    for index in range(start_idx, end_idx + 1):
        try:
            # generate questions
            get_answer(index, generator, gen_modelId, embedder)
        except Exception or RuntimeError as err:
            if error_questions < ERROR_THRESHOLD:
                error_questions += 1
                # Get the full stack trace as a string
                stack_trace = traceback.format_exc()
                logging.info(
                    f"Question of index: {index} cannot be processed, adding to DLQ. "
                    f"{ERROR_THRESHOLD - error_questions} tries left. error: {err}\nStack trace:\n{stack_trace}"
                )
                error_entry = {"num": index, "err": str(err), "trace": stack_trace}
                DLQ.append(error_entry)
                write_to_dlq(dlq_path, str(error_entry))
            else:
                logging.error(f"Error threshold reached. Question after index {index} will not be processed.")
                DLQ.append({"num": index, "err": err})
                print("DLQ: ", DLQ)
                write_to_dlq(dlq_path, str({"num": index, "err": err}))
                raise err
            
    end_time = time.time()
    time_taken = time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))
    logging.info(f"done! time to evaluate {end_idx - start_idx + 1} trees: {time_taken}")
    print(f"done! time to evaluate {end_idx - start_idx + 1} trees: {time_taken}")
    print("DLQ: ", DLQ)