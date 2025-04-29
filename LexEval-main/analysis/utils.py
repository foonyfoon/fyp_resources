import utils.constants as constants
from tree.tree import ReadTree
from tree.node import RootNode, SemanticNode, SyntacticNode

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from collections import deque
import json

gen_models = ["google/gemma-3-12b-it", "google/gemma-3-1b-it", "mistral.mistral-7b-instruct-v0:2"]

agg_funcs = {
    'base_found_rag_match': ['sum', lambda x: (~x).sum()],  # sum of Trues, sum of Falses
    'base_found_match': ['sum', lambda x: (~x).sum()],
    'fk_score': ['mean', 'var'],
    'dc_score': ['mean', 'var'],
    'root_similarity_score': ['mean', 'var'],
}

# Define base colors for models
model_colors = {
    'google/gemma-3-1b-it': 'green',
    'google/gemma-3-12b-it': 'blue',
    'mistral.mistral-7b-instruct-v0:2': 'red',
}


def read_trees_to_df(strategy: str, gen_modelId: str) -> pd.DataFrame:
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
    strategy_path = constants.STRATEGY_PATH_DICT[strategy]
    tree_dir_path = f"/vol/bitbucket/lst20/treenodes/{strategy_path}/gemma3-12b_perturb/3_2_0/{gen_modelId.replace('/', '-')}/complete/"

    rows = []
    df = pd.read_csv(constants.SHUFFLED_FILE)
    for _, row in df.iterrows():
        i = row['original_index']
        try:
            tree_path = f"{tree_dir_path}{i}_checked.pkl"
            tree = ReadTree.load_read_tree(tree_path)
            root = tree.root
            possible_answers = tree.possible_answers
            queue = deque([(root, 0)])  # (node, layer)
            visited = {root}
            question_id = i

            while queue:
                # evaluate answer
                node, layer = queue.popleft()
                base_found_match = False
                base_found_rag_match = False
                base_response = node.answers[gen_modelId]["base"]
                base_rag_response = node.answers[gen_modelId]["base_rag"]
                for expected_answer in json.loads(possible_answers):
                    if base_response.__contains__(expected_answer):
                        base_found_match = True
                        break
                for expected_answer in json.loads(possible_answers):
                    if base_rag_response.__contains__(expected_answer):
                        base_found_rag_match = True
                        break
                    
                node_data = {
                    "gen_modelId": gen_modelId,
                    "question_id": question_id,
                    "layer": layer,
                    "id": node.id,
                    "type": node.__class__.__name__,
                    "prompt": node.prompt,
                    "rag_closest_match": getattr(node, "rag_closest_match", None),
                    "rag_entities": getattr(node, "rag_entities", None),
                    "answers": node.answers,
                    "wiki_title": getattr(node, "wiki_title", None),
                    "base_found_rag_match": base_found_rag_match, 
                    "base_found_match": base_found_match, 
                }

                if isinstance(node, (RootNode, SemanticNode)):
                    node_data.update({
                        "root_similarity_score": getattr(node, "root_similarity_score", None),
                        "complexity_score": getattr(node, "complexity_score", None),
                        "fk_score": getattr(node, "fk_score", None),
                        "dc_score": getattr(node, "dc_score", None),
                    })

                elif isinstance(node, (SyntacticNode)):
                    node_data["syntax_similarity_score"] = getattr(node, "syntax_similarity_score", None)

                rows.append(node_data)

                for child in node.children:
                    if child not in visited:
                        queue.append((child, layer + 1))
                        visited.add(child)

        except FileNotFoundError:
            # print(f"Tree file not found at {tree_path}")
            pass
        except Exception as e:
            print(f"Error loading tree: {e}")

    return pd.DataFrame(rows)


def plot_by_layer(df, gen_models, title):
    # Utility to lighten a color towards white
    def lighten_color(color, amount=0.5):
        try:
            c = mcolors.cnames[color]
        except KeyError:
            c = color
        c = mcolors.to_rgb(c)
        return tuple(x + (1.0 - x) * amount for x in c)
    
    fig, ax = plt.subplots(figsize=(10, 6))

    for model in gen_models:
        df_model = df[df['gen_modelId'] == model].copy()

        # Calculate percentages
        df_model['base_match_percentage'] = df_model["base_found_match_True_count"] / (
            df_model["base_found_match_True_count"] + df_model["base_found_match_False_count"]
        ) * 100
        df_model['base_rag_match_percentage'] = df_model["base_found_rag_match_True_count"] / (
            df_model["base_found_rag_match_True_count"] + df_model["base_found_rag_match_False_count"]
        ) * 100

        # Colors
        base_color = model_colors.get(model, 'black')
        rag_color = lighten_color(base_color, amount=0.5)

        # Plot base
        ax.plot(df_model['layer'], df_model['base_match_percentage'], 
                marker='o', color=base_color, label=f'{model.split("/")[-1]} - base')
        
        # Plot base_rag
        ax.plot(df_model['layer'], df_model['base_rag_match_percentage'], 
                marker='x', linestyle='--', color=rag_color, label=f'{model.split("/")[-1]} - base_rag')

    ax.set_ylabel('Correct Answer Percentage (%)')
    ax.set_xlabel('Layer')
    ax.set_title(f'{title} Correct Answer Percentage over Layers')
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


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
