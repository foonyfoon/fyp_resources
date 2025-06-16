# LexEval
We built upon LexEval (by Siddhant Singh) to run on local models and allow testing for 3 types of perturbation to prompts: paraphrasing, dialect translation and adding extraneous information with positional perturbation.

## Setup and Installation
Install required packages from: `pip install -r requirements.txt`

## Additional Notes
You must have HuggingFace membership to run experiments. AWS credentials as well, if you intend to evaluate mdels hosted there. However for final evalutions all models are run locally and AWS models are only used for initial experimentation. Feel free to remove any unused code in the `model/engine.py` section. Else follow https://docs.aws.amazon.com/cli/v1/userguide/cli-chap-configure.html to set up the credentials.

## Usage
1. run experiments:
see usage in `prefix-3_2_0.py` for non-terminal perturbation and `apply_term.py` for terminal perturbation.
1. read results:
call `read_trees_to_df()` in `analysis/utils.py` to export tree to DataFrame for analysis.
Example for getting paraphrased trees:
## Credits
Siddhant Singh: for the original LexEval LLM evaluation framework
HuggingFace: for the adapted perplexity calculation code in `analysis/utils.py
