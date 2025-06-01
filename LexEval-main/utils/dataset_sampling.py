import os
import re
import logging


import utils.constants as constants
from adapters.OAI_Embeddings import EmbedAdapter, RobertaEmbedder

from sklearn.model_selection import train_test_split
import spacy
from typing import List, Tuple
from datasets import load_dataset, concatenate_datasets
import torch
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

nlp = spacy.load("en_core_web_trf")


'''
Both PopQA and TriviaQA (TQA) include Wikipedia references. The columns we use are:
- "question": question string
- "possible_answers": answer string in list format
- "s_wiki_title" : Wikipedia title
- "original_index": queistion index of original dataset

Note: If using a different dataset, adapt these column names and formats accordingly.
'''

def read_dataset(name, size, columns=None):
    if name == "POPQA":
        return read_popqa_dataset(size, columns)
    elif name == "TQA":
        return read_tqa_dataset(size, columns)
    else:
        raise ValueError(f"read dataset {name} not implemented")


def read_popqa_dataset(size, columns=None):
    if not os.path.exists(constants.SHUFFLED_FILE):
        if not os.path.exists(constants.POPQA_DF_LOCATION):
            # Ensure the directory exists
            dir = os.path.dirname(constants.POPQA_DF_LOCATION)
            if not os.path.exists(dir):
                os.makedirs(dir)

            # Read original dataset
            df = pd.read_csv("hf://datasets/akariasai/PopQA/test.tsv", sep="\t")
            logging.info("read from hf source")
        else:
            # Load already saved dataset
            df = pd.read_parquet(constants.POPQA_DF_LOCATION)
            logging.info("read from local source")

        # Ensure the directory exists
        dir = os.path.dirname(constants.SHUFFLED_FILE)
        if not os.path.exists(dir):
            os.makedirs(dir)

        # Add original index as a column
        df["original_index"] = df.index

        # Shuffle dataset with original index column preserved
        # df_shuffled = df.sample(frac=1, random_state=seed)
        size_ratio = size / len(df)
        # Split the data to get a subset with the same distribution of classes
        _, df_shuffled, _, _ = train_test_split(
            df,
            df["prop"],
            test_size=size_ratio,
            shuffle=True,
            random_state=constants.SEED,
            stratify=df["prop"],
        )
        # Save shuffled dataset
        df_shuffled.to_parquet(constants.SHUFFLED_FILE, index=False)

        logging.info(f"df_shuffled saved to {constants.SHUFFLED_FILE}")

    # Load already saved shuffled dataset
    if columns is None:
        df_shuffled = pd.read_parquet(constants.SHUFFLED_FILE)
    else:
        df_shuffled = pd.read_parquet(constants.SHUFFLED_FILE, columns=columns)

    return df_shuffled


def read_tqa_dataset(size, columns=None):
    if not os.path.exists(constants.SHUFFLED_TQA_FILE):
        if not os.path.exists(constants.TQA_LABELLED_DF_LOCATION):
            # 1. load cluster labelled dataset from constants.TQA_labelled; throw error if not exist
            tqa_dataset= _load_wiki_rc_triviaqa()
            tqa_df = tqa_dataset.to_pandas()
            # 2. format it to be popQA like
            dataset = pd.DataFrame()
            dataset['possible_answers'] = tqa_df['answer'].apply(lambda x: x['value'])
            dataset['s_wiki_title'] = tqa_df['entity_pages'].apply(lambda x: re.findall(r"'([^']*)'", ( x['title'])))
            dataset['question'] = tqa_df['question']
            # Drop rows where any of the columns has NaN
            dataset.dropna(subset=['possible_answers', 's_wiki_title', 'question'], inplace=True)
            # add cluster label
            _add_class(dataset) # class col = cluster_id
            # Ensure the directory exists
            dir = os.path.dirname(constants.TQA_LABELLED_DF_LOCATION)
            if not os.path.exists(dir):
                os.makedirs(dir)
            dataset.to_parquet(constants.TQA_LABELLED_DF_LOCATION)
            
        else: 
            tqa_dataset = pd.read_parquet(constants.TQA_LABELLED_DF_LOCATION)
        
        # 3. stratify dataset based on cluster labels
        size_ratio = size / len(dataset)
        # Split the data to get a subset with the same distribution of classes
        dataset["original_index"] = dataset.index
        _, df_shuffled, _, _ = train_test_split(
            dataset,
            dataset["cluster_id"],
            test_size=size_ratio,
            shuffle=True,
            random_state=constants.SEED,
            stratify=dataset["cluster_id"],
        )
        # Ensure the directory exists
        out_dir = os.path.dirname(constants.SHUFFLED_TQA_FILE)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        # Save shuffled dataset
        df_shuffled.to_parquet(constants.SHUFFLED_TQA_FILE, index=False)

        logging.info(f"tqa_df_shuffled saved to {constants.SHUFFLED_TQA_FILE}")
    
    # Load already saved shuffled dataset
    if columns is None:
        df_shuffled = pd.read_parquet(constants.SHUFFLED_TQA_FILE)
    else:
        df_shuffled = pd.read_parquet(constants.SHUFFLED_TQA_FILE, columns=columns)

    return df_shuffled


def _load_wiki_rc_triviaqa():
    # 1) load the reading‐comprehension config
    ds = load_dataset("mandarjoshi/trivia_qa", "rc")
    wiki_train = ds["train"].filter(lambda ex:
        len(ex["entity_pages"]['doc_source']) > 0 and
        len(ex["entity_pages"]['doc_source']) <= 3)
    wiki_dev = ds["validation"].filter(lambda ex:
        len(ex["entity_pages"]['doc_source']) > 0 and
        len(ex["entity_pages"]['doc_source']) <= 3)
    wiki_test = ds["test"].filter(lambda ex:
        len(ex["entity_pages"]['doc_source']) > 0 and
        len(ex["entity_pages"]['doc_source']) <= 3)

    wiki_full = concatenate_datasets([wiki_train, wiki_dev, wiki_test])
    
    seen_qids = set()
    keep_indices = []
    for idx, example in enumerate(wiki_full):
        qid = example["question_id"]
        if qid not in seen_qids:
            seen_qids.add(qid)
            keep_indices.append(idx)
    wiki_unique = wiki_full.select(keep_indices)
    return wiki_unique


def _add_class(df: pd.DataFrame) -> None:
    embedder= RobertaEmbedder()
    masked_embeddings, _ = _mask_entities_and_embed_batch(
        df['question'], embedder, mask_token='<mask>', mask_entities=True, batch_size=256
    )
    cluster_idx = _cluster_and_assign_grp(masked_embeddings, df['question'], k=25)
    df["cluster_id"] = cluster_idx['cluster_id']
    
    
def _mask_entities_and_embed_batch(
    prompts: List[str],
    embedder: EmbedAdapter,
    mask_token: str = "<mask>",
    mask_entities: bool = True,
    batch_size: int = 256,
) -> Tuple[torch.Tensor, List[List[str]]]:
    """
    Finds named entities in each prompt, optionally replaces them all with mask_token,
    embeds the (masked or original) prompts in batches, and returns:
      - a tensor of shape (len(prompts), hidden_size)
      - a list of lists of entity strings found in each prompt

    Args:
      prompts: list of input strings
      embedder: an instance of EmbedAdapter
      mask_token: token to replace each entity with
      mask_entities: if False, skip masking and embed original prompts
      batch_size: number of prompts to process per batch
    """
    all_embeddings: List[torch.Tensor] = []
    all_entities: List[List[str]] = []

    # Process prompts in chunks
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        masked_batch: List[str] = []
        batch_entities: List[List[str]] = []

        for prompt in batch:
            doc = nlp(prompt)
            # Collect entity texts
            entities = [ent.text for ent in doc.ents]
            # Mask entities if requested
            if mask_entities and entities:
                masked = prompt
                for ent in sorted(doc.ents, key=lambda e: e.start_char, reverse=True):
                    masked = masked[: ent.start_char] + mask_token + masked[ent.end_char :]
            else:
                masked = prompt

            masked_batch.append(masked)
            batch_entities.append(entities)

        # Embed the (masked or original) batch
        embeddings = embedder.encode(masked_batch)
        all_embeddings.append(embeddings)
        all_entities.extend(batch_entities)

    # Concatenate all batch embeddings
    if all_embeddings:
        result_embeddings = torch.cat(all_embeddings, dim=0)
    else:
        # No prompts: return empty tensor of shape (0, hidden_size)
        result_embeddings = torch.empty((0,))

    return result_embeddings, all_entities


def _cluster_and_assign_grp(
    embeddings: torch.Tensor,
    prompts: List[str],
    k: int = 25,
):
    # Step 1: Run KMeans on CPU
    kmeans = KMeans(n_clusters=k, random_state=constants.SEED)
    cluster_ids = kmeans.fit_predict(embeddings.cpu().numpy())
    # Prepare DataFrame for return
    df = pd.DataFrame({
        "prompt": prompts,
        "cluster_id": cluster_ids
    })
    return df