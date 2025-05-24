EVAL_SIZE = 1000
SEED = 42
RETRY_COUNT = 0
ERROR_THRESHOLD = 700

################ DIR ################
POPQA_DF_LOCATION = "/vol/bitbucket/lst20/lex-eval_dataset/PopQA/test.parquet"
SHUFFLED_FILE = "/vol/bitbucket/lst20/lex-eval_dataset/PopQA/shuffled.parquet"

TQA_LABELLED_DF_LOCATION = "/vol/bitbucket/lst20/lex-eval_dataset/TQA/labelled.parquet"
SHUFFLED_TQA_FILE = "/vol/bitbucket/lst20/lex-eval_dataset/TQA/shuffled.parquet"

TREE_DIR = "/vol/bitbucket/lst20/"

LOG_DIR = "/vol/bitbucket/lst20/logs/"
DLQ_DIR = "/vol/bitbucket/lst20/dlq/"
TIMER_DIR = '/vol/bitbucket/lst20/timers/'
GRAPH_DIR = '/vol/bitbucket/lst20/graph/'
#####################################

STRATEGY_PATH_DICT = {
    "prefix":"prefix",
    "paraphrase":"para",
    "paraphrase_then_prefix":"para-prefix",
    }

TREE_SIZE = (3, 2, 0)

MODELS = [
    'google/gemma-3-1b-it',
    'google/gemma-3-12b-it',
    # 'mistral.mistral-7b-instruct-v0:2',
    'mistralai/Mistral-7B-Instruct-v0.2',
    ]

dataset_fancy_text = {
    'POPQA': 
    '''
____ _____ _____ _____ _____ 
|  _  |     |  _  |     |  _  |
|   __|  |  |   __|  |  |     |
|__|  |_____|__|  |__  _|__|__|
                    |__|                                    
    ''',
    'TQA':
    '''            
 _____ _____ _____ 
|_   _|     |  _  |
  | | |  |  |     |
  |_| |__  _|__|__|
         |__|      
    '''
}