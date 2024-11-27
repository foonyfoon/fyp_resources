# LexEval
 With LexEval, we have created a scalable evaluation framework adaptable to the continuous advancements in language models. It enables users to fine-tune their evaluation paradigms, focusing on syntactic or semantic perturbations as required.


## Installation
To run LexEval, you will need some specific libraries and assets, run this command in the terminal in the root directory:

`pip install -r requirements.txt`

## Additional Notes

You must have OpenAI, TogetherAI and HuggingFace memberships along with enough credit in them to perform experiments (we recommend ~$10 in each as a starting point).
The environment variables are as follows:
- OpenAI: `OPENAI_API_KEY`
- Together AI: `TOGETHER_API_KEY`

### Model Names:
When using LexEval, please use the model names defined by each model provider, they are case-sensitive and the functions will fail if the `model_name` string is malformed.
- OpenAI Models: https://platform.openai.com/docs/models/gpt-4-turbo-and-gpt-4 (`MODEL` column)
- Together AI Models: https://docs.together.ai/docs/inference-models (`Model String for API` column)

### Dataset imports:
The code for POPQA has already been written in `PopQA.py` but to run LexEval on other datasets, you will need to write your own dataset adapters. Note: ensure you have a pro HuggingFace subscription.

## Usage
1. Import the necessary files:
```python
from tree.node import RootNode, SemanticNode, SyntacticNode
from tree.tree import Tree
from adapters.SemanticAdapter import SemanticAdapter
from adapters.SyntacticPerturb import SyntacticPerturber
```

2. Initialise the semantic and syntactic adapters:
```python
semantic_adapter = SemanticAdapter()
syntactic_adapter = SyntacticPerturber()
```

3. Create the tree object
```python
tree = Tree(
    "<Your prompt here>",
    semantic_adapter,
    syntactic_adapter
)
```

4. Invoke the `make_tree` function to build the tree, `depth`, `num_semantic` and `num_syntactic` are hyperparameters you can change (Note: exceeding `depth` more than `3` may result in severe slowdown of the creation process and may cost > $3 per tree. We recommend `<3,3,2>` as the maximum configuration).
```python
tree.make_tree(
    depth=2, 
    num_semantic=2, 
    num_syntactic=2, 
    model_name='gpt-3.5-turbo'
    )
```

5. Once your tree has been built, you can set the `possible_answers` attribute of the tree before you invoke the checking function, note you should wrap the list around `' '`.
```python
tree.set_possible_answers('[<strings of your expected answers>]')
```

6. After building your tree and setting the possible answers, you can run the batched checking function to invoke LexEval.
For example, with `google/gemma-7b-it`:
```python
tree.run_check_pop_qa_batched(
    model_name="google/gemma-7b-it", 
    batch_size=5
    )
```
Note: you can run this multiple times with the same tree with different models and all the answers will be stored in the tree itself.

You can also compute BLEU and ROUGE metrics as follows:
```python
tree.add_bleu_and_rouge(
    model_name='google/gemma-7b-it'
    )
```

7. Print out the tree in terminal.
```python
tree.print_tree()
```

## Running on a dataset
You can run LexEval on a dataset and use `pickle` to store the tree in your directory, for example:
```python

# Making and storing the tree
for (question, possible_answers, row_id) in dataset:
    tree = Tree(
        question,
        semantic_adapter,
        syntactic_adapter
    )
    tree.set_possible_answers(possible_answers)
    
    tree.make_tree(
        depth=2, 
        num_semantic=2, 
        num_syntactic=2, 
        model_name='gpt-3.5-turbo'
    )
    with open(f'{row_id}.pkl', 'wb') as f:
        pickle.dump(tree, f)

# Retrieving and running the check
for (_, _, row_id) in dataset:
    tree = pickle.load(open(f'{row_id}.pkl'))

    # Run eval function with batch size of 3 (recommended)
    tree.run_check_pop_qa_batched(
        model_name="google/gemma-7b-it", 
        batch_size=3
    )

    # Add BLEU and ROUGE metrics
    tree.add_bleu_and_rouge(
        model_name='google/gemma-7b-it'
    )
     
    # Store the tree back with the answers from the evaluation
    with open(f'{row_id}.pkl', 'wb') as f:
        pickle.dump(tree, f)
```


## Credit
Author: Siddhant Singh