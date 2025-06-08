from tree.node import TerminalNode
import utils.constants as constants
from tree.tree import ReadTree
from tree.node import RootNode, SemanticNode, SyntacticNode
from utils.dataset_sampling import read_dataset

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import colorsys
import torch
import numpy as np
import evaluate
from evaluate import logging
from torch.nn import CrossEntropyLoss
import datasets

from collections import deque
import json
import re
import os
scorer = rouge_scorer.RougeScorer(['rougeL', 'rouge1'], use_stemmer=True)
smooth_fn = SmoothingFunction().method1

gen_models = ["google/gemma-3-12b-it", "google/gemma-3-1b-it", "mistral.mistral-7b-instruct-v0:2"]


# Define base colors for models
model_colors = {
    'google/gemma-3-1b-it': 'green',
    'google/gemma-3-12b-it': 'blue',
    'mistral.mistral-7b-instruct-v0:2': 'red',
}

def read_trees_to_df(strategy: str, gen_modelId: str, dataset: str, terminal=False, long=False) -> pd.DataFrame:
    '''
    read trees from perturb strategy and generator to pandas df
    columns:
    - question_id (from tree path)
    ----- all nodes -----
    - layer
    - id
    - type
    - prompt
    - rag_closest_match
    - rag_entities
    - answers
    - wiki_title
    --- root/sem node ---
    - root_similarity_score
    - complexity_score
    - fk_score
    - dc_score
    ----- syn node ------
    - syntax_similarity_score
    ---------------------
    '''

    def extract_similarity(meta):
        if isinstance(meta, dict) and 'similarity' in meta:
            return meta['similarity']
        return 1
    long_prefix = "long_" if long else ""
    strategy_path = constants.STRATEGY_PATH_DICT[strategy]
    tree_dir_path = f"/vol/bitbucket/lst20/{long_prefix}{dataset}_treenodes/{strategy_path}/gemma3-12b_perturb/3_2_0/{gen_modelId.replace('/', '-')}/complete/"
    print(tree_dir_path)
    if not os.path.isdir(tree_dir_path) or not os.listdir(tree_dir_path):
        return pd.DataFrame()
    rows = []
    df = read_dataset(dataset, 1000)
    for _, row in df.iterrows():
        i = row['original_index']
        try:
            tree_path = f"{tree_dir_path}{i}_checked.pkl"
            tree = ReadTree.load_read_tree(tree_path)
            root = tree.root
            possible_answers = tree.possible_answers

            # ensure possible_answers is a Python list of strings
            if isinstance(possible_answers, str):
                try:
                    parsed = json.loads(possible_answers)
                    possible_answers = parsed if isinstance(parsed, list) else [parsed]
                except json.JSONDecodeError:
                    possible_answers = [possible_answers]
            elif not isinstance(possible_answers, list):
                possible_answers = [possible_answers]

            queue = deque([(root, 0)])  # (node, layer)
            visited = {root}
            question_id = i

            while queue:
                base_found_match = False
                rag_found_match = False
                node, layer = queue.popleft()

                # grab your two generated outputs
                if not node.answers:
                    continue
                if not terminal and isinstance(node, TerminalNode):
                    continue
                base_response = node.answers[gen_modelId]["base"].lower()
                rag_response = node.answers[gen_modelId]["base_rag"].lower()

                # initialize best-F1 trackers
                best = {k:-1 for k in [
                    'base_f1_1','base_f1_L','rag_f1_1','rag_f1_L',
                    'base_rec_1','base_rec_L','rag_rec_1','rag_rec_L',
                    'base_prec_1','base_prec_L','rag_prec_1','rag_prec_L',
                    # 'base_bleu','rag_bleu'
                ]}

                # for each gold answer, compute rouge-L
                for exp in possible_answers:
                    exp_low = str(exp).lower()
                    exp_low = re.sub(r'[^\w\s]', '', exp_low)

                    # 1) simple containment flags:
                    if exp_low in base_response:
                        base_found_match = True
                    if exp_low in rag_response:
                        rag_found_match = True
                    
                    # 2) Rouge-L F1
                    scores_b  = scorer.score(exp_low, base_response)
                    scores_br = scorer.score(exp_low, rag_response)
                    metrics = {
                        'base_f1_1':scores_b['rouge1'].fmeasure,
                        'base_f1_L':scores_b['rougeL'].fmeasure,
                        'rag_f1_1':scores_br['rouge1'].fmeasure,
                        'rag_f1_L':scores_br['rougeL'].fmeasure,
                        'base_rec_1':scores_b['rouge1'].recall,
                        'base_rec_L':scores_b['rougeL'].recall,
                        'rag_rec_1':scores_br['rouge1'].recall,
                        'rag_rec_L':scores_br['rougeL'].recall,
                        'base_prec_1':scores_b['rouge1'].precision,
                        'base_prec_L':scores_b['rougeL'].precision,
                        'rag_prec_1':scores_br['rouge1'].precision,
                        'rag_prec_L':scores_br['rougeL'].precision,
                    }
                    # BLEU
                    # metrics['base_bleu'] = sentence_bleu([exp_low.split()], base_response.split(), smoothing_function=smooth_fn)
                    # metrics['rag_bleu']  = sentence_bleu([exp_low.split()], rag_response.split(), smoothing_function=smooth_fn)

                    for k,v in metrics.items():
                        if v > best[k]:
                            best[k] = v
        
                # find gen em source
                source = getattr(node, "rag_closest_match", None)
                
                if source:                   
                    # terminal node type
                    if isinstance(node, TerminalNode):
                        para_type = node.metadata["terminal_name"]
                        recorded_layer = layer - 1
                    else:
                        para_type = "semantic"
                        recorded_layer = layer
                    node_data = {
                        "gen_modelId": gen_modelId,
                        "question_id": question_id,
                        "layer": recorded_layer,
                        "id": node.id,
                        "type": node.__class__.__name__ ,
                        "para_type": para_type,
                        "prompt": node.prompt,
                        "rag_closest_match": getattr(node, "rag_closest_match", None),
                        "rag_entities": getattr(node, "rag_entities", None),
                        "possible_answers": possible_answers,
                        "answers": node.answers,
                        "metadata": node.metadata,
                        "wiki_title": getattr(node, "wiki_title", None),
                        "rag_found_match": rag_found_match, 
                        "base_found_match": base_found_match,
                        **best
                    }

                    if isinstance(node, (RootNode, SemanticNode)):
                        node_data.update({
                            "root_similarity_score": getattr(node, "root_similarity_score", 0),
                            "complexity_score": getattr(node, "complexity_score", 0),
                            "fk_score": getattr(node, "fk_score", 0),
                            "dc_score": getattr(node, "dc_score", 0),
                        })

                    elif isinstance(node, (SyntacticNode)):
                        node_data["syntax_similarity_score"] = getattr(node, "syntax_similarity_score", None)

                    rows.append(node_data)
                
                for child in node.children:
                    if child not in visited:
                        queue.append((child, layer + 1))
                        visited.add(child)

        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Error loading tree: {e}")
    df = pd.DataFrame(rows)
    
    if df.get('metadata') is not None:
        df['similarity'] = df.get('metadata').apply(extract_similarity)
    else:
        df['similarity'] = 1
    # Convert nested dicts into a DataFrame
    base_rag_df = df['answers'].apply(lambda d: list(d.values())[0] if isinstance(d, dict) else {}).apply(pd.Series)

    # Merge back into the original DataFrame
    df = pd.concat([df, base_rag_df[['base', 'base_rag']]], axis=1)
    df = df.drop(columns=['answers'])

    return df
 
gen_models = ["google/gemma-3-12b-it", "google/gemma-3-1b-it", "mistralai/Mistral-7B-Instruct-v0.2"]
#, "mistral.mistral-7b-instruct-v0:2"
# Define base colors for models

model_colors = {
    'google/gemma-3-1b-it':'green',  # green
    'google/gemma-3-12b-it': 'blue',  # blue
    # 'mistral.mistral-7b-instruct-v0:2': 'red',  # red
    'mistralai/Mistral-7B-Instruct-v0.2': 'black',  # black
}


def agg_df(df, group_by=['gen_modelId','para_type','layer']):
    return (
        df.groupby(group_by)
          .agg(
            base_true      = ('base_found_match','sum'),
            base_false     = ('base_found_match',lambda x:(~x).sum()),
            base_pct       = ('base_found_match',lambda x:x.mean()*100),
            base_rag_true  = ('rag_found_match','sum'),
            base_rag_false = ('rag_found_match',lambda x:(~x).sum()),
            base_rag_pct   = ('rag_found_match',lambda x:x.mean()*100),
            base_f1_1      = ('base_f1_1','mean'),
            rag_f1_1       = ('rag_f1_1','mean'),
            base_f1_L      = ('base_f1_L','mean'),
            rag_f1_L       = ('rag_f1_L','mean'),
            base_rec_1     = ('base_rec_1','mean'),
            rag_rec_1      = ('rag_rec_1','mean'),
            base_rec_L     = ('base_rec_L','mean'),
            rag_rec_L      = ('rag_rec_L','mean'),
            base_prec_1    = ('base_prec_1','mean'),
            rag_prec_1     = ('rag_prec_1','mean'),
            base_prec_L    = ('base_prec_L','mean'),
            rag_prec_L     = ('rag_prec_L','mean'),
          )
          .reset_index()
    )
def plot_by_layer(
    agg_df,
    gen_models,
    title,
    terminal_comparison: list = [],
    plot_base: bool = True,
    plot: str = "percent"    # options: "percent", "f1_ROUGE_L", "f1_ROUGE_1", "recall_ROUGE_L", "recall_ROUGE_1", "prec_ROUGE_L", "prec_ROUGE_1", "bleu"
):
    fig, ax = plt.subplots(figsize=(10,6))

    # Determine which columns to plot based on 'plot' parameter
    if plot == "f1_ROUGE_L":
        ycol_base, ycol_rag = 'base_f1_L', 'rag_f1_L'
        ylabel = 'Average Rouge-L F1 Score'
    elif plot == "f1_ROUGE_1":
        ycol_base, ycol_rag = 'base_f1_1', 'rag_f1_1'
        ylabel = 'Average Rouge-1 F1 Score'
    elif plot == "recall_ROUGE_L":
        ycol_base, ycol_rag = 'base_rec_L', 'rag_rec_L'
        ylabel = 'Average Rouge-L Recall'
    elif plot == "recall_ROUGE_1":
        ycol_base, ycol_rag = 'base_rec_1', 'rag_rec_1'
        ylabel = 'Average Rouge-1 Recall'
    elif plot == "prec_ROUGE_L":
        ycol_base, ycol_rag = 'base_prec_L', 'rag_prec_L'
        ylabel = 'Average Rouge-L Precision'
    elif plot == "prec_ROUGE_1":
        ycol_base, ycol_rag = 'base_prec_1', 'rag_prec_1'
        ylabel = 'Average Rouge-1 Precision'
    elif plot == "bleu":
        ycol_base, ycol_rag = 'base_bleu', 'rag_bleu'
        ylabel = 'Average BLEU Score'
    else:
        ycol_base, ycol_rag = 'base_pct', 'base_rag_pct'
        ylabel = 'EM Percentage (%)'

    for model in gen_models:
        dfm = agg_df.query("gen_modelId == @model and para_type == 'semantic'")
        color = model_colors.get(model, 'black')
        name  = model.split("/")[-1]

        if plot_base:
            ax.plot(dfm['layer'], dfm[ycol_base], marker='o', color=color, label=f'{name} - base')
        ax.plot(dfm['layer'], dfm[ycol_rag], marker='x', linestyle='--', color=color, label=f'{name} - base_rag')
       
        term_markers = ["d", "H", "v","s", "p", "+", "1", "."]
        for i, term in enumerate(terminal_comparison):
            dft = agg_df.query("gen_modelId == @model and para_type == @term")
            if dft.empty:
                continue
            ax.plot(dft['layer'], dft[ycol_rag], marker=term_markers[i], linestyle='-.', color=color, label=f'{name} – {term}')

    ax.set_xlabel('Layer')
    ax.set_ylabel(ylabel)
    title_map = {
        'f1_ROUGE_L': 'Rouge-L F1', 'f1_ROUGE_1': 'Rouge-1 F1',
        'recall_ROUGE_L': 'Rouge-L Recall', 'recall_ROUGE_1': 'Rouge-1 Recall',
        'prec_ROUGE_L': 'Rouge-L Precision', 'prec_ROUGE_1': 'Rouge-1 Precision',
        'percent': 'EM', 'bleu': 'BLEU'
    }
    title_txt = title_map.get(plot, 'Accuracy')
    ax.set_title(f'{title} over Layers ({title_txt})')
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def get_perplexity(df):
    # Initialize perplexity metric
    metric = Perplexity()

    # Placeholder for scores
    perplexity_scores = []

    # Group by model and compute perplexity for each text
    for model_id, group in df.groupby("gen_modelId"):
        texts = group["prompt"].tolist()

        # Compute perplexities for this model
        results = metric._compute(
            predictions=texts,
            model_id=model_id,
        )

        # Assign per-example perplexities back to the group
        group_perplexities = results["perplexities"]

        # Append to list (preserve original index for merging)
        perplexity_scores.extend(zip(group.index, group_perplexities))

    # Create a new DataFrame from scores and merge back
    score_df = pd.DataFrame(perplexity_scores, columns=["index", "perplexity"])
    score_df.set_index("index", inplace=True)
    return score_df



## HAVEN'T CHECKED FUNCTIONS FROM HERE BEYOND!!
def generate_base_rag_comparison(
    df: pd.DataFrame,
    by_layer: bool = False,
) -> pd.DataFrame:
    """
    Compute counts of how questions changed from base→RAG.
    """
    df_temp = df.copy()
    df_temp['deteriorated']     = ((df_temp['base_found_match']) & (~df_temp['base_found_rag_match'])).astype(int)
    df_temp['improved']         = ((~df_temp['base_found_match']) & (df_temp['base_found_rag_match'])).astype(int)
    df_temp['stable_correct']   = ((df_temp['base_found_match']) & (df_temp['base_found_rag_match'])).astype(int)
    df_temp['stable_incorrect'] = ((~df_temp['base_found_match']) & (~df_temp['base_found_rag_match'])).astype(int)

    group_cols = []
    group_cols.append('gen_modelId')
    if by_layer:
        group_cols.append('layer')

    if group_cols:
        return (
            df_temp
            .groupby(group_cols)[
                ['deteriorated', 'improved', 'stable_correct', 'stable_incorrect']
            ]
            .sum()
            .reset_index()
        )
    else:
        sums = df_temp[['deteriorated', 'improved', 'stable_correct', 'stable_incorrect']].sum()
        return pd.DataFrame([sums])


def plot_models_by_layer(
    stats_df: pd.DataFrame,
    title: str,
    model_colors: dict,
    normalized: bool = False
):
    """
    Plots the four RAG‐change categories in a 1×4 grid of bar charts.

    """
    cats = ['deteriorated','improved','stable_correct','stable_incorrect']
    titles = {
        'deteriorated':     'Deteriorated\n(correct→incorrect)',
        'improved':         'Improved\n(incorrect→correct)',
        'stable_correct':   'Stable Correct\n(correct→correct)',
        'stable_incorrect': 'Stable Incorrect\n(incorrect→incorrect)'
    }
    
    df = stats_df.copy()
    if normalized:
        # convert each category to percent of total per row
        total = df[cats].sum(axis=1)
        df[cats] = df[cats].div(total, axis=0) * 100

    models = list(df['gen_modelId'].unique())
    n_models = len(models)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5), sharey=True)
    
    for ax, cat in zip(axes, cats):
        # pivot so rows=layer, cols=model
        pivot = df.pivot(index='layer', columns='gen_modelId', values=cat)
        layers = pivot.index.astype(str).tolist()
        x = np.arange(len(layers))
        total_width = 0.8
        bar_width = total_width / n_models

        for i, model in enumerate(models):
            ys = pivot.get(model, pd.Series(0, index=pivot.index)).values
            color = model_colors.get(model, None)
            ax.bar(
                x + i*bar_width,
                ys,
                bar_width,
                label=model.split('/')[-1],
                color=color
            )

        ax.set_xticks(x + total_width/2 - bar_width/2)
        ax.set_xticklabels(layers, rotation=45, ha='right')
        ax.set_title(titles[cat])
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        if normalized:
            ax.set_ylabel('Percentage of Questions (%)')
        else:
            ax.set_ylabel('Count of Questions')

    # shared legend (only once)
    axes[-1].legend(title='Model', loc='upper right', fontsize='small')
    fig.suptitle(f'RAG Change fro {title} by Model & Layer', fontsize=18)
    plt.tight_layout()
    plt.show()


def get_model(modelId: str) :
    if "gemma-3" in modelId.lower():
        from model.engine import Gemma3Adapter
        adapter: Gemma3Adapter = Gemma3Adapter(modelId)
    elif "gemma" in modelId.lower():
        from model.engine import GemmaAdapter
        adapter: GemmaAdapter = GemmaAdapter(modelId)
    elif "mistral" in modelId.lower():
        from model.engine import MistralInstructAdapter
        adapter = MistralInstructAdapter('mistralai/Mistral-7B-Instruct-v0.2')
    else:
        raise NotImplementedError(f"No adapter implemented for: {modelId}")
    return adapter.model, adapter.tokenizer, adapter.device


class Perplexity(evaluate.Metric):
    # Adapted from the original Hugging Face Evaluate Perplexity metric implementation:
    # https://github.com/huggingface/evaluate/blob/main/metrics/perplexity/perplexity.py
    #
    # Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
    # Licensed under the Apache License, Version 2.0.
    #
    # This adaptation allows integration with custom model loading logic and eval the quantized version
    def _info(self):
        return evaluate.MetricInfo(
            module_type="metric",
            description="_DESCRIPTION",
            citation="_CITATION",
            inputs_description="_KWARGS_DESCRIPTION",
            features=datasets.Features(
                {
                    "predictions": datasets.Value("string"),
                }
            ),
            reference_urls=["https://huggingface.co/docs/transformers/perplexity"],
        )

    def _compute(
            self, predictions, model_id, batch_size: int = 8, add_start_token: bool = True, max_length=None
        ):
            model, tokenizer, device = get_model(model_id)
            model = model.to(device)
            
            PAD_MULTIPLE = 8
            if max_length:
                max_length = (max_length // PAD_MULTIPLE) * PAD_MULTIPLE
            # if batch_size > 1 (which generally leads to padding being required), and
            # if there is not an already assigned pad_token, assign an existing
            # special token to also be the padding token
            if tokenizer.pad_token is None and batch_size > 1:
                existing_special_tokens = list(tokenizer.special_tokens_map_extended.values())
                # check that the model already has at least one special token defined
                assert (
                    len(existing_special_tokens) > 0
                ), "If batch_size > 1, model must have at least one special token to use for padding. Please use a different model or set batch_size=1."
                # assign one of the special tokens to also be the pad token
                tokenizer.add_special_tokens({"pad_token": existing_special_tokens[0]})

            if add_start_token and max_length:
                # leave room for <BOS> token to be added:
                assert (
                    tokenizer.bos_token is not None
                ), "Input model must already have a BOS token if using add_start_token=True. Please use a different model, or set add_start_token=False"
                max_tokenized_len = max_length - 1
            else:
                max_tokenized_len = max_length

            encodings = tokenizer(
                predictions,
                pad_to_multiple_of=PAD_MULTIPLE,
                padding=True,
                max_length=max_tokenized_len,
                return_tensors="pt",
                return_attention_mask=True,
            ).to(device)

            encoded_texts = encodings["input_ids"]
            attn_masks = encodings["attention_mask"]

            # check that each input is long enough:
            if add_start_token:
                assert torch.all(torch.ge(attn_masks.sum(1), 1)), "Each input text must be at least one token long."
            else:
                assert torch.all(
                    torch.ge(attn_masks.sum(1), 2)
                ), "When add_start_token=False, each input text must be at least two tokens long. Run with add_start_token=True if inputting strings of only one token, and remove all empty input strings."

            ppls = []
            loss_fct = CrossEntropyLoss(reduction="none")

            for start_index in logging.tqdm(range(0, len(encoded_texts), batch_size)):
                end_index = min(start_index + batch_size, len(encoded_texts))
                encoded_batch = encoded_texts[start_index:end_index]
                attn_mask = attn_masks[start_index:end_index]

                if add_start_token:
                    bos_tokens_tensor = torch.tensor([[tokenizer.bos_token_id]] * encoded_batch.size(dim=0)).to(device)
                    encoded_batch = torch.cat([bos_tokens_tensor, encoded_batch], dim=1)
                    attn_mask = torch.cat(
                        [torch.ones(bos_tokens_tensor.size(), dtype=torch.int64).to(device), attn_mask], dim=1
                    )

                labels = encoded_batch

                with torch.no_grad():
                    out_logits = model(encoded_batch, attention_mask=attn_mask).logits

                shift_logits = out_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                shift_attention_mask_batch = attn_mask[..., 1:].contiguous()

                perplexity_batch = torch.exp(
                    (loss_fct(shift_logits.transpose(1, 2), shift_labels) * shift_attention_mask_batch).sum(1)
                    / shift_attention_mask_batch.sum(1)
                )

                ppls += perplexity_batch.tolist()

            return {"perplexities": ppls, "mean_perplexity": np.mean(ppls)}