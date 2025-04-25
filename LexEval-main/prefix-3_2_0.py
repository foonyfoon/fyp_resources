from tree.tree import Tree
from adapters.SemanticAdapter import SemanticAdapter
from adapters.SemanticPerturb import (
    SemanticPerturber,
    ParaphrasePerturber,
    PrefixPerturber,
    CombinedPerturber,
)
from adapters.SyntacticPerturb import SyntacticPerturber
from adapters.OAI_Embeddings import EmbedAdapter, RobertaEmbedder
from model.engine import LLMAdapter, Gemma3Adapter
from utils.timer import Timers
from wiki_cache.db import engine, Session
import wiki_cache.models as models
from wiki_cache.cache import clear_cache_db

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

SEED = 42
RETRY_COUNT = 0
ERROR_THRESHOLD = 500
DLQ = []  # Dead Letter Queue
DF_LOCATION = "/vol/bitbucket/lst20/lex-eval_dataset/PopQA/test.csv"
SHUFFLED_FILE = "/vol/bitbucket/lst20/lex-eval_dataset/PopQA/shuffled.csv"

# prefix, paraphrase, paraphrase_then_prefix
STRATEGY = "paraphrase"
# "para-prefix" prefix para
statrgy_path = "para"
intermediatory_tree_dir = f"/vol/bitbucket/lst20/treenodes/{statrgy_path}/gemma3-12b_perturb/3_2_0/tree"
checked_tree_dir = f"/vol/bitbucket/lst20/treenodes/{statrgy_path}/gemma3-12b_perturb/3_2_0/gemma-3-12b/complete"
TIMER_PATH = f"/vol/bitbucket/lst20/timers/gemma3-12b-with-caching-{statrgy_path}.pkl"


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
    torch.cuda.synchronize()


def configure_logging(filename):
    dir = os.path.dirname(filename)
    if not os.path.exists(dir):
        os.makedirs(dir)
    # Configure logging
    logging.basicConfig(
        filename=filename,
        filemode="a",
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )


def write_to_dlq(filename, text):
    with open(filename, "a") as dlq_file:
        dlq_file.write(f"{text}:\n")


def get_answer(index, og_index, generator, modelId, embedder):
    tree_file_path = f"{intermediatory_tree_dir}/{og_index}.pkl"
    checked_tree_file_path = f"{checked_tree_dir}/{og_index}_checked.pkl"
    max_retries = RETRY_COUNT
    for retry_count in range(max_retries + 1):
        logging.info(f"Evaluating tree of dataset row: {index}, attempt: {retry_count}")
        try:
            # Load tree in evaluation mode using the generator
            test_tree = Tree.load_tree(
                tree_file_path, eval=True, generator=generator, embedder=embedder
            )
            test_tree.run_check_pop_qa_batched(index=index, model_name=modelId)
            # test_tree.add_bleu_and_rouge(model_name=modelId)
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
                logging.info(
                    f"After clearing cache: {free_mem}/{total_mem} memory available"
                )
            else:
                raise err  # re-raise if it’s a different ValueError

        except RuntimeError as err:
            logging.error(f"RuntimeError encountered on attempt {retry_count}: {err}")
            if retry_count == max_retries:
                raise err  # re-raise if max retries reached


def get_question_tree(
    index, row, semantic_adapter, syntactic_adapter, ner_model, embedder, tree_size
):
    max_retries = RETRY_COUNT
    for retry_count in range(max_retries + 1):
        try:
            logging.info(
                f"creating tree of dataset row: {index}, attempt: {retry_count}"
            )
            question = row["question"]
            possible_answers = row["possible_answers"]
            s_uri = row["s_uri"]
            s_uri_code = s_uri.rstrip("/").split("/")[-1]
            og_index = row["original_index"]

            if not os.path.exists("trees/"):
                os.makedirs("trees/")
            tree_file_path = f"{intermediatory_tree_dir}/{og_index}.pkl"

            # Only create a tree if not already saved
            if not os.path.exists(tree_file_path):
                test_tree = Tree(
                    question,
                    eval=False,
                    sem_perturber=semantic_adapter,
                    syn_perturber=syntactic_adapter,
                    ner_model=ner_model,
                    embedder=embedder,
                    s_uri_code=s_uri_code,
                )
                test_tree.make_tree(*tree_size, index=index)
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
                logging.info(
                    f"After clearing cache: {free_mem}/{total_mem} memory available"
                )
            else:
                raise err
        except RuntimeError as err:
            logging.error(f"RuntimeError encountered on attempt {retry_count}: {err}")
            if retry_count == max_retries:
                raise err


def read_dataset(seed, columns=None):
    if not os.path.exists(SHUFFLED_FILE):
        if not os.path.exists(DF_LOCATION):
            # Ensure the directory exists
            dir = os.path.dirname(DF_LOCATION)
            if not os.path.exists(dir):
                os.makedirs(dir)

            # Read original dataset
            df = pd.read_csv("hf://datasets/akariasai/PopQA/test.tsv", sep="\t")
            print("read from hf source")
        else:
            # Load already saved dataset
            df = pd.read_csv(DF_LOCATION)
            print("read from local source")

        # Ensure the directory exists
        dir = os.path.dirname(SHUFFLED_FILE)
        if not os.path.exists(dir):
            os.makedirs(dir)

        # Add original index as a column
        df["original_index"] = df.index

        # Shuffle dataset with original index column preserved
        df_shuffled = df.sample(frac=1, random_state=seed)

        # Save shuffled dataset
        df_shuffled.to_csv(SHUFFLED_FILE, index=False)
        print(f"df_shuffled saved to {SHUFFLED_FILE}")

    # Load already saved shuffled dataset
    if columns is None:
        df_shuffled = pd.read_csv(SHUFFLED_FILE)
    else:
        df_shuffled = pd.read_csv(SHUFFLED_FILE, usecols=columns)

    return df_shuffled


def select_perturber(type: str, params: dict) -> SemanticPerturber:
    """
    Factory method to select and instantiate a SemanticPerturber.
    """
    def get_paraphraser(paraphrase_modelId: str, embedder: EmbedAdapter) -> ParaphrasePerturber:
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
        return ParaphrasePerturber(model, embedder)

    type = type.lower()
    
    if type == "prefix":
        return PrefixPerturber(params["embedder"], params["tree_size"])
    
    elif type == "paraphrase":
        return get_paraphraser(params["modelId"], params["embedder"])
     
    elif type == "paraphrase_then_prefix":   
        paraphraser = get_paraphraser(params["modelId"], params["embedder"])
        prefixer = PrefixPerturber(params["embedder"], params["tree_size"])
        return CombinedPerturber([paraphraser, prefixer])
    else:
        raise ValueError(f"Unknown perturber type: {type}")

if __name__ == "__main__":
    # Check if the correct number of arguments is passed
    if len(sys.argv) < 3:
        print("wrong params!", sys.argv)
        sys.exit(1)
    formatted_time = datetime.fromtimestamp(time.time()).strftime("%m-%d-%H:%M:%S")
    filename = f"/vol/bitbucket/lst20/logs/{formatted_time}_logs.log"
    configure_logging(filename)

    # create dlq file
    dlq_path = f"/vol/bitbucket/lst20/dlq/{formatted_time}_dlq.log"
    dlq_dir = os.path.dirname(dlq_path)
    if not os.path.exists(dlq_dir):
        os.makedirs(dlq_dir)
    with open(dlq_path, "a") as dlq_file:
        dlq_file.write("DLQ:\n")

    start_idx = int(sys.argv[1])
    end_idx = int(sys.argv[2])
    current_index = start_idx
    
    Timers.reset()

    logging.info(f"processing dataset from {start_idx} to {end_idx}")

    gc.enable()

    error_questions = 0

    # create table
    models.Base.metadata.drop_all(engine)
    models.Base.metadata.create_all(bind=engine)

    start_time = time.time()
    logging.info(
        "Start time: %s", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    )
    ##################### factory this ###########################
    tree_size = (3, 2, 0)
    syntactic_adapter = SyntacticPerturber()
    embedder = RobertaEmbedder()
    ##############################################################
    model_provisioned_time = time.time()
    model_start_duration = model_provisioned_time - start_time
    logging.info(f"model ready in: {model_start_duration:.3g} s")
    logging.info(f"intermediatory_tree_dir: {intermediatory_tree_dir} s")
    df = read_dataset(
        SEED, columns=["question", "possible_answers", "s_uri", "original_index"]
    )
    # generate questions
    params = {
        "modelId": "google/gemma-3-12b-it",
        "embedder": embedder,
        "tree_size": tree_size
    }
    with select_perturber(STRATEGY, params) as semantic_adapter:
        for index, row in df.iloc[start_idx : end_idx + 1].iterrows():
            try:
                with torch.no_grad():
                    get_question_tree(
                        index,
                        row,
                        semantic_adapter,
                        syntactic_adapter,
                        None,
                        embedder,
                        tree_size,
                    )
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
                    logging.error(
                        f"Error threshold reached. Question after index {index} will not be processed."
                    )
                    DLQ.append({"num": index, "err": err})
                    print("DLQ: ", DLQ)
                    write_to_dlq(dlq_path, str({"num": index, "err": err}))
                    raise err
            # Clear cache after processing sample
            clear_cache_db()

    # post perturb cleanup
    Session.remove()
    engine.dispose()
    clear_cache()

    free_mem, total_mem = torch.cuda.mem_get_info()
    logging.info(f"after clearing cache, {free_mem / 1024**2:.2f} MB/{total_mem / 1024**2:.2f} MB memory available")
    logging.info(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
    logging.info(f"Cached:    {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

    logging.info("********* start eval tree *********")
    # #############################################################
    gen_modelId = "google/gemma-3-12b-it"
    generator_model = Gemma3Adapter(gen_modelId)
    generator = generator_model
    embedder = RobertaEmbedder()
    # ############################################################
    for index, row in df.iloc[start_idx : end_idx + 1].iterrows():
        try:
            # generate questions
            og_index = row["original_index"]
            get_answer(index, og_index, generator, gen_modelId, embedder)
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
                logging.error(
                    f"Error threshold reached. Question after index {index} will not be processed."
                )
                DLQ.append({"num": index, "err": err})
                print("DLQ: ", DLQ)
                write_to_dlq(dlq_path, str({"num": index, "err": err}))
                raise err

    Timers.print_report()
    Timers.save(TIMER_PATH)

    end_time = time.time()
    time_taken = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))
    logging.info(
        f"done! time to evaluate {end_idx - start_idx + 1} trees: {time_taken}"
    )
    print(f"done! time to evaluate {end_idx - start_idx + 1} trees: {time_taken}")
    print("DLQ: ", DLQ)
