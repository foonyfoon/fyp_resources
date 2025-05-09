from tree.tree import Tree
from adapters.SemanticPerturb import (
    SemanticPerturber,
    ParaphrasePerturber,
    PrefixPerturber,
    CombinedPerturber,
)
from adapters.SyntacticPerturb import SyntacticPerturber
from adapters.OAI_Embeddings import EmbedAdapter, RobertaEmbedder
from model.engine import LLMAdapter
from utils.timer import Timers
from wiki_cache.db import engine, Session
import wiki_cache.models as models
from wiki_cache.cache import clear_cache_db
from utils.dataset_sampling import read_dataset

import torch

import os
import logging
import time
from datetime import datetime
import gc
import traceback
import argparse
import utils.constants as constants

# ####################### DEFAULT #######################
# prefix, paraphrase, paraphrase_then_prefix
strategy = "prefix"
# google/gemma-3-12b-it mistral.mistral-7b-instruct-v0:2
gen_modelId = "google/gemma-3-12b-it"
eval_only = False
# #######################################################
statrgy_path = constants.STRATEGY_PATH_DICT[strategy]
intermediatory_tree_dir = f"/vol/bitbucket/lst20/treenodes/{statrgy_path}/gemma3-12b_perturb/3_2_0/tree"
checked_tree_dir = f"/vol/bitbucket/lst20/treenodes/{statrgy_path}/gemma3-12b_perturb/3_2_0/{gen_modelId.replace('/', '-')}/complete"
timer_path = f"/vol/bitbucket/lst20/timers/gemma3-12b-with-caching-{statrgy_path}.pkl"
DLQ = []  # Dead Letter Queue


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


def get_answer(index, og_index, device, generator, modelId, embedder):
    tree_file_path = f"{intermediatory_tree_dir}/{og_index}.pkl"
    checked_tree_file_path = f"{checked_tree_dir}/{og_index}_checked.pkl"
    max_retries = constants.RETRY_COUNT
    for retry_count in range(max_retries + 1):
        logging.info(f"Evaluating tree of dataset row: {index}, attempt: {retry_count}")
        try:
            if not os.path.exists(checked_tree_file_path):
                test_tree = Tree.load_tree(
                    device, tree_file_path, eval=True, generator=generator, embedder=embedder
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
    max_retries = constants.RETRY_COUNT
    for retry_count in range(max_retries + 1):
        try:
            logging.info(
                f"creating tree of dataset row: {index}, attempt: {retry_count}"
            )
            question = row["question"]
            possible_answers = row["possible_answers"]
            s_wiki_title = row["s_wiki_title"]
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
                    s_wiki_title=s_wiki_title,
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


def select_perturber(type: str, params: dict) -> SemanticPerturber:
    """
    Factory method to select and instantiate a SemanticPerturber.
    """
    if eval_only:
        return None
    def get_paraphraser(paraphrase_modelId: str, embedder: EmbedAdapter) -> ParaphrasePerturber:
        from model.engine import Gemma3Adapter
        model = Gemma3Adapter(paraphrase_modelId)
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


def get_generator(modelId: str) -> ParaphrasePerturber:
    model: LLMAdapter = None
    if "gemma-3" in modelId.lower():
        from model.engine import Gemma3Adapter
        model = Gemma3Adapter(modelId)
    elif "gemma" in modelId.lower():
        from model.engine import GemmaAdapter
        model = GemmaAdapter(modelId)
    elif "mistralai" in modelId.lower():
        from model.engine import MistralInstructAdapter
        model = MistralInstructAdapter(modelId)
    elif "mistral.mistral-7b-instruct-v0:2" in modelId.lower():
        from model.engine import MistralInstructAwsAdapter
        model = MistralInstructAwsAdapter(modelId)
    else:
        raise NotImplementedError(f"No adapter implemented for: {modelId}")
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="Script with strategy and model flags, plus 2 required integers + dataset name.")
    
    # Optional named flags
    parser.add_argument("--strategy", type=str, default=strategy, help="Strategy to use (e.g., prefix, para-prefix)")
    parser.add_argument("--gen_modelId", type=str, default=gen_modelId, help="Generation model ID (e.g., google/gemma-3-12b-it)")
    parser.add_argument("--eval_only", action="store_true", help="Flag to run in evaluation-only mode.")

    # Compulsory positional integers and dataset name
    parser.add_argument("start_idx", type=int, help="starting index for processing of dataset.")
    parser.add_argument("end_idx", type=int, help="ending index for processing of dataset.")
    parser.add_argument("dataset_name", type=str, help="questions to answer (POPQA, TQA).")
    
    return parser.parse_args()


def update_params():
    global statrgy_path, intermediatory_tree_dir, checked_tree_dir, timer_path
    statrgy_path = constants.STRATEGY_PATH_DICT[strategy]
    intermediatory_tree_dir = f"/vol/bitbucket/lst20/treenodes/{statrgy_path}/gemma3-12b_perturb/3_2_0/tree"
    checked_tree_dir = f"/vol/bitbucket/lst20/treenodes/{statrgy_path}/gemma3-12b_perturb/3_2_0/{gen_modelId.replace('/', '-')}/complete"
    timer_path = f"/vol/bitbucket/lst20/timers/gemma3-12b-with-caching-{statrgy_path}.pkl"


def main():
    # ###################### parse input ######################
    global strategy, gen_modelId, eval_only
    args = parse_args()
    # prefix, paraphrase, paraphrase_then_prefix
    strategy = args.strategy
    gen_modelId = args.gen_modelId
    eval_only = args.eval_only
    start_idx = args.start_idx
    end_idx = args.end_idx
    dataset_name = args.dataset_name
    update_params()
     
    formatted_time = datetime.fromtimestamp(time.time()).strftime("%m-%d-%H:%M")
    filename = f"/vol/bitbucket/lst20/logs/{formatted_time}_{start_idx}_to_{end_idx}_{statrgy_path}.log"
    configure_logging(filename)

    # create dlq file
    dlq_path = f"/vol/bitbucket/lst20/dlq/{formatted_time}_dlq.log"
    dlq_dir = os.path.dirname(dlq_path)
    if not os.path.exists(dlq_dir):
        os.makedirs(dlq_dir)
    with open(dlq_path, "a") as dlq_file:
        dlq_file.write("DLQ:\n")
    
    Timers.reset()

    logging.info(f"processing dataset from {start_idx} to {end_idx}")

    gc.enable()

    error_questions = 0

    # create table
    models.Base.metadata.drop_all(engine)
    models.Base.metadata.create_all(bind=engine)

    logging.info(f"intermediatory_tree_dir: {intermediatory_tree_dir}")
    logging.info(f"answer generator: {gen_modelId}")
    start_time = time.time()
        
    df = read_dataset(
        name=dataset_name,
        size=constants.EVAL_SIZE,
        columns=["question", "possible_answers", "s_wiki_title", "original_index"]
    )
    
    if not eval_only:
        logging.info(
            "Start time: %s", time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
        )
        syntactic_adapter = SyntacticPerturber()
        embedder = RobertaEmbedder()
        model_provisioned_time = time.time()
        model_start_duration = model_provisioned_time - start_time
        logging.info(f"model ready in: {model_start_duration:.3g} s")
        
        # generate questions
        params = {
            "modelId": "google/gemma-3-12b-it",
            "embedder": embedder,
            "tree_size": constants.TREE_SIZE
        }
        with select_perturber(strategy, params) as semantic_adapter:
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
                            constants.TREE_SIZE,
                        )
                except Exception or RuntimeError as err:
                    if error_questions < constants.ERROR_THRESHOLD:
                        error_questions += 1
                        # Get the full stack trace as a string
                        stack_trace = traceback.format_exc()
                        logging.info(
                            f"Question of index: {index} cannot be processed, adding to DLQ. "
                            f"{constants.ERROR_THRESHOLD - error_questions} tries left. error: {err}\nStack trace:\n{stack_trace}"
                        )
                        error_entry = {"num": index, "err": str(err), "trace": stack_trace}
                        DLQ.append(error_entry)
                        write_to_dlq(dlq_path, str(error_entry))
                    else:
                        logging.error(
                            f"Error threshold reached. Question after index {index} will not be processed."
                        )
                        DLQ.append({"num": index, "err": err})
                        logging.info("DLQ: ", DLQ)
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
    generator = get_generator(gen_modelId)
    embedder = RobertaEmbedder()
    device=generator.device 

    for index, row in df.iloc[start_idx : end_idx + 1].iterrows():
        try:
            # generate questions
            og_index = row["original_index"]
            get_answer(index, og_index, device, generator, gen_modelId, embedder)
        except Exception or RuntimeError as err:
            if error_questions < constants.ERROR_THRESHOLD:
                error_questions += 1
                # Get the full stack trace as a string
                stack_trace = traceback.format_exc()
                logging.info(
                    f"Question of index: {index} cannot be processed, adding to DLQ. "
                    f"{constants.ERROR_THRESHOLD - error_questions} tries left. error: {err}\nStack trace:\n{stack_trace}"
                )
                error_entry = {"num": index, "err": str(err), "trace": stack_trace}
                DLQ.append(error_entry)
                write_to_dlq(dlq_path, str(error_entry))
            else:
                logging.error(
                    f"Error threshold reached. Question after index {index} will not be processed."
                )
                DLQ.append({"num": index, "err": err})
                logging.info("DLQ: ", DLQ)
                write_to_dlq(dlq_path, str({"num": index, "err": err}))
                raise err

    Timers.print_report()
    Timers.save(timer_path)

    end_time = time.time()
    time_taken = time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))
    logging.info(
        f"done! time to evaluate {end_idx - start_idx + 1} trees: {time_taken}"
    )


# example: prefic-3_2_0 10 150 --strategy=prefix --eval_only --gen_modelId=google/gemma-3-1b-it
if __name__ == "__main__":
    main()