from tree.tree import Tree
from adapters.SemanticAdapter import SemanticAdapter
from adapters.SyntacticPerturb import SyntacticPerturber
from model.engine import ClaudeAdapter
from adapters.rag import RAGAgent

import pandas as pd

import os
import logging
import time
import gc
import sys
import math

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)

RETRY_COUNT = 5
ERROR_THRESHOLD = 8
DLQ = []  # Dead Letter Queue
CHUNK_SIZE = 50
modelId = 'anthropic.claude-3-haiku-20240307-v1:0'
   
def process_with_retry(retry_count, index, row, model, semantic_adapter, syntactic_adapter, rag):
    try:
        logging.info(f"Processing row with index: {index}, attempt: {retry_count}")
        question = row["question"]
        possible_answers = row["possible_answers"]
        
        tree_file_path = f"/vol/bitbucket/lst20/treenodes/claude_2_2_1/tree/{index}.pkl"
        checked_tree_file_path = f"/vol/bitbucket/lst20/treenodes/claude_2_2_1/complete/{index}_checked.pkl"
        # if exist file load
        test_tree = None
        if not os.path.exists(tree_file_path):
            test_tree = Tree(
                question,
                semantic_adapter,
                syntactic_adapter,
                rag
            )
            test_tree.make_tree(2, 2, 1, model_name='anthropic.claude-3-haiku-20240307-v1:0')
            test_tree.save_tree(tree_file_path)
        test_tree = Tree.load_tree(tree_file_path, semantic_adapter, syntactic_adapter, rag)
        test_tree.set_possible_answers(possible_answers)
        test_tree.run_check_pop_qa_batched(model_name='anthropic.claude-3-haiku-20240307-v1:0', batch_size=5)
        test_tree.add_bleu_and_rouge(
            model_name='anthropic.claude-3-haiku-20240307-v1:0'
            )
        test_tree.save_tree(checked_tree_file_path)
        test_tree.print_tree()
    except RuntimeError as err:
        logging.error(f"RuntimeError encountered on attempt {retry_count}: {err}")
        if 'ThrottlingException' in str(err):
            raise err
        elif retry_count >= RETRY_COUNT:
            raise err
        #  delete model and re-instantiate
        else:
            del model
            model = ClaudeAdapter(modelId)
            semantic_adapter = SemanticAdapter(model)
            syntactic_adapter = SyntacticPerturber()
            rag = RAGAgent(model)
            process_with_retry(retry_count + 1, index, row, model, semantic_adapter, syntactic_adapter, rag)

if __name__ == "__main__":
    # Check if the correct number of arguments is passed
    if len(sys.argv) < 3:
        print("wrong params!", sys.argv)
        sys.exit(1)

    start_idx = int(sys.argv[1])
    end_idx = int(sys.argv[2])
    current_index = start_idx

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
    
    model = ClaudeAdapter(modelId)
    semantic_adapter = SemanticAdapter(model)
    syntactic_adapter = SyntacticPerturber()
    rag = RAGAgent(model)
    client_provisioned_time = time.time()
    client_start_duration = client_provisioned_time - start_time
    logging.info(f"client provisioned in: {client_start_duration:.3g} s")
    
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
    time_taken = time.strftime('%H:%M:%S', time.gmtime(end_time - start_time))
    logging.info(f"done! time to evaluate {end_idx - start_idx + 1} trees: {time_taken}")
    print(f"done! time to evaluate {end_idx - start_idx + 1} trees: {time_taken}")
    logging.info(f"token usage: in_token={model.input_tokens}, out_token={model.output_tokens}")
    
    # pricing
    input_price_per_1k = 0.0008
    output_price_per_1k = 0.004
    input_token_cost = math.ceil(model.input_tokens / 1000) * input_price_per_1k
    output_token_cost = math.ceil(model.output_tokens / 1000) * output_price_per_1k
    total_cost = input_token_cost + output_token_cost
    logging.info(f"price estimation: ${input_token_cost:.3g} + ${output_token_cost:.3g} = $USD {total_cost:.3g}")   
    print(f"price estimation: ${input_token_cost:.3g} + ${output_token_cost:.3g} = $USD {total_cost:.3g}")   
