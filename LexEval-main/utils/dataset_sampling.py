import os
import utils.constants as constants
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

'''
Both PopQA and TriviaQA (TQA) include Wikipedia references. The columns we use are:
- `question`: the question text  
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
            df = pd.read_csv(constants.POPQA_DF_LOCATION)
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
        df_shuffled.to_csv(constants.SHUFFLED_FILE, index=False)

        logging.info(f"df_shuffled saved to {constants.SHUFFLED_FILE}")

    # Load already saved shuffled dataset
    if columns is None:
        df_shuffled = pd.read_csv(constants.SHUFFLED_FILE)
    else:
        df_shuffled = pd.read_csv(constants.SHUFFLED_FILE, usecols=columns)

    return df_shuffled


def read_tqa_dataset(size, columns=None):
    '''
    stratify class is based on ....
    since questions are ....
    '''
    # 1. load cluster labelled dataset from constants.TQA_labelled; throw error if not exist
    # 2. stratify dataset based on cluster labels
    # 3. format it to be popQA like
    # ['question', 'question_id', 'question_source', 'entity_pages', 'search_results', 'answer']
    
    
    pass